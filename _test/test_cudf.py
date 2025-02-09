import cudf
import cugraph
import time
import dask_cudf

start = time.time()
# file_path = '../_datasets/combined_dataset_sample_st.csv'
# gdf = cudf.read_csv(file_path, delimiter=',')
file_path = '../_datasets/combined_dataset_st.parquet'
gdf = cudf.read_parquet(file_path)
# gdf = dask_cudf.from_cudf(cudf.read_parquet(file_path))
end = time.time()
print(f'##### read file: {end - start:.5f} sec')

print(gdf)

G = cugraph.MultiGraph()
G.from_cudf_edgelist(gdf, source='source', destination='target')

print(G.number_of_nodes(), G.number_of_edges())

deg = cugraph.degree_centrality(G)
print(deg)

