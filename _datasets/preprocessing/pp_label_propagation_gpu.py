import cudf
import cugraph
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import cupy as cp
import json
import time
import numpy as np

def create_gpu_graph(file_path):
    """
    대규모 금융 거래 네트워크를 GPU에서 생성하는 함수 (cuGraph 활용)

    :param file_path: 데이터 파일 경로 (.parquet 또는 .csv)
    :return: GPU 기반 cuGraph 그래프 객체 및 문자열 ID 매핑 정보
    """

    # 🚀 데이터 로드 (Dask 기반 대용량 데이터 처리)
    if file_path.endswith(".parquet"):
        ddf = dask_cudf.read_parquet(file_path, split_out=10)  # 🔥 병렬 로딩 추가
    elif file_path.endswith(".csv"):
        ddf = dask_cudf.read_csv(file_path, dtype={"source": "str", "target": "str"}, split_out=10)
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. CSV 또는 Parquet을 사용하세요.")

    print(f"✅ Data Loaded: {file_path}")

    # 🚀 필요한 컬럼 선택 (source, target, weight)
    ddf = ddf[["source", "target", "amount"]]

    # 🚀 💡 source, target을 int64로 변환 (Factorization)
    unique_nodes = cudf.concat([ddf["source"], ddf["target"]]).unique()
    node_map = cudf.Series(data=cp.arange(len(unique_nodes), dtype="int64"), index=unique_nodes)

    # 🚀 문자열을 정수 ID로 매핑
    ddf["source"] = ddf["source"].map(node_map).astype("int64")
    ddf["target"] = ddf["target"].map(node_map).astype("int64")

    print("✅ String IDs Converted to Int64 IDs")

    # 🚀 중복 간선을 그룹화하여 Weight 처리 (Multi-Edges → Weighted Edge)
    ddf = ddf.groupby(["source", "target"]).agg({"amount": "sum"}).reset_index()

    # 🚀 GPU 기반 cudf 변환
    gdf = ddf.compute().to_cudf()
    print(f"✅ Data Converted to cuDF (GPU)")

    # 🚀 cuGraph 무방향 그래프 생성
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source="source", destination="target", edge_attr="amount")

    print(f"✅ Graph Created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, gdf, node_map


import cudf
import cugraph
import cupy as cp
import json
import time


def propagation_laundering_score(G, gdf, node_map, max_iter=10):
    """
    GPU 기반 Louvain 커뮤니티 탐색 및 Label Propagation 수행

    :param G: cuGraph 그래프 객체
    :param gdf: cuDF 데이터 프레임 (source, target, amount)
    :param node_map: {str → int64} 변환된 노드 매핑 정보
    :param max_iter: Label Propagation 최대 반복 횟수
    :return: laundering_score (각 노드의 원래 문자열 값 매핑 결과)
    """

    # 🚀 1️⃣ Louvain 커뮤니티 탐색 (GPU)
    start = time.time()
    communities = cugraph.louvain(G)
    end = time.time()
    print(f"✅ Louvain Communities: {len(communities)} detected in {end - start:.5f} sec")

    # 🚀 2️⃣ 노드 ID를 원래 `str` 값으로 변환 (역매핑)
    inverse_node_map = {v: k for k, v in node_map.to_pandas().items()}

    communities_dict = communities.to_pandas().set_index("vertex")["partition"].to_dict()
    communities_str = {inverse_node_map[node]: community for node, community in communities_dict.items()}

    # 🚀 3️⃣ Louvain 커뮤니티 결과 JSON 저장 (문자열 노드 ID로 변환)
    with open("../communities_cd_v0.1.json", "w") as f:
        json.dump(communities_str, f, indent=4)

    print(f"✅ Saved Louvain Communities with Original Node Names")

    # 🚀 4️⃣ Score 초기화 (cuDF 사용)
    laundering_score = cudf.Series(data=cp.zeros(G.number_of_nodes()), index=communities["vertex"].to_pandas())

    # 🚀 5️⃣ 자금세탁 거래 노드 초기값 설정
    laundering_nodes = gdf[gdf["amount"] > 0]  # 금액이 있는 경우로 가정 (변경 가능)
    laundering_score.loc[laundering_nodes["source"]] = 1
    laundering_score.loc[laundering_nodes["target"]] = 1

    print(f"✅ Initialized laundering scores.")

    # 🚀 6️⃣ Propagation (CuPy 활용한 병렬 연산)
    for i in range(max_iter):
        start = time.time()

        # 모든 노드의 이웃 평균값 업데이트 (Jaccard Similarity 기반)
        neighbors_df = cugraph.jaccard(G).to_pandas()
        neighbors_df["score"] = neighbors_df["jaccard_coeff"] * laundering_score.loc[neighbors_df["destination"]].values

        # 노드별 평균 계산
        new_scores = neighbors_df.groupby("source")["score"].mean().to_dict()

        # 업데이트
        laundering_score = laundering_score.to_pandas()
        laundering_score.update(new_scores)
        laundering_score = cudf.Series(laundering_score)

        end = time.time()
        print(f"✅ Iteration {i + 1}: Label Propagation complete in {end - start:.5f} sec")

    # 🚀 7️⃣ Label Propagation 결과를 원래 `str` 값으로 변환
    laundering_score_dict = laundering_score.to_pandas().to_dict()
    laundering_score_str = {inverse_node_map[node]: score for node, score in laundering_score_dict.items()}

    return laundering_score_str


if __name__ == "__main__":
    start = time.time()
    file_path = "../combined_dataset_v0.1.parquet"

    # GPU 기반 Graph 생성
    G, gdf, node_map = create_gpu_graph(file_path)

    # Louvain + Label Propagation 수행
    laundering_score = propagation_laundering_score(G, gdf, node_map)

    # 저장
    with open("../laundering_score_cd_v0.1.json", "w") as f:
        json.dump(laundering_score, f)

    end = time.time()
    print(f"Completed all processes in {end - start:.5f} sec")
