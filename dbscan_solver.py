import numpy as np
import pandas as pd
from helper import Helper
import matplotlib.pyplot as plt


######################################################################
####Author: Risav Pokhrel
####Github: https://github.com/Risav25Pokhrel
######################################################################

class DbScanSolver:
    
    def __init__(self,path):
        datapath=self.hp.convertTxToCSV(path=path,fileName="dataset1")
        self.df=pd.read_csv(datapath[0])
        self.numberOfVehicles=datapath[2]
        self.vehicleCapacity=datapath[1]
        self.customerPositions=self.df.iloc[:,[1,2]].to_numpy()
        self.numberOfCustomers=len(self.customerPositions)
        print("Data set Loaded")
        plt.scatter(self.customerPositions[:,0],self.customerPositions[:,1],s=10, c= "black")
        plt.title("Customer Positions")
        plt.show()
    
    customerPositions=None
    vehicleCapacity=None
    numberOfVehicles=None
    numberOfCustomers=None
    df=None

    hp=Helper()

###########################################~~~~~DBSCAN SOLVER~~~~~#################################################
    def dbScan(self, min_radius: float, max_radius: float, max_cluster_size_const: int):
        final_cluster = None
        best_average_cluster_size = 0
        best_radius = 0
        
        # Store original bounds for debugging
        original_min = min_radius
        original_max = max_radius
        
        while min_radius < max_radius:
            radius = (min_radius + max_radius) / 2
            clusters = self.hp.DBScan(self.customerPositions, radius, min_samples=5)
            
            # Handle case where no clusters are formed (all points are noise)
            unique_clusters = np.unique(clusters)
            if len(unique_clusters) == 1 and unique_clusters[0] == -1:  # Only noise points
                max_radius = radius-1
                print(f"\nNo clusters formed at radius {radius}, reducing max_radius to {max_radius}")
                continue
                
            cluster_counts = np.unique(clusters, return_counts=True)[1]
            # Filter out noise points (cluster label -1) when calculating max size
            if -1 in unique_clusters:
                noise_idx = np.where(unique_clusters == -1)[0][0]
                cluster_counts = np.delete(cluster_counts, noise_idx)
            
            if len(cluster_counts) == 0:  # All points are noise
                max_radius = radius-1
                print(f"\nAll points are noise at radius {radius}, reducing max_radius")
                continue
                
            max_cluster_size = np.max(cluster_counts)
            num_actual_clusters = len(cluster_counts)  # Exclude noise
            
            print(f"Radius: {radius:.3f}, Max cluster size: {max_cluster_size}, Num clusters: {num_actual_clusters}")
            
            if max_cluster_size > max_cluster_size_const:
                max_radius = radius-1
                print(f"Max cluster size {max_cluster_size} exceeds limit {max_cluster_size_const}, reducing max_radius to {max_radius}")
            else:
                # Valid clustering found
                average_cluster_size = self.numberOfCustomers / num_actual_clusters
                
                if average_cluster_size > best_average_cluster_size:
                    best_average_cluster_size = average_cluster_size
                    final_cluster = clusters.copy()
                    best_radius = radius
                    print(f"\nNew best solution: radius {radius:.3f}, avg cluster size {average_cluster_size:.2f}")
                
                min_radius = radius+1
                print(f"Valid clustering found, increasing min_radius to {min_radius}\n")
            
            # Convergence check to prevent infinite loops
            if abs(max_radius - min_radius) < 0.001:  # Small epsilon
                print(f"Converged: min_radius={min_radius:.3f}, max_radius={max_radius:.3f}")
                break
        
        if final_cluster is None:
            print(f"No Solution Found...")
            print(f"Searched radius range: [{original_min:.3f}, {original_max:.3f}]")
            print(f"Final radius: {radius:.3f}")
            print("Try adjusting parameters: increase max_cluster_size_const or adjust radius range")
            return None
        
        print("\n~~~~~~~~~~~Cluster Formed~~~~~~~~~~~\n")
        print(f"Best radius: {best_radius:.3f} with average cluster size: {int(best_average_cluster_size)}\n")
        print(f"Checking Capacity Constraints, vehicle capacity: {self.vehicleCapacity}\n")
        
        # Work with a copy to avoid modifying the original DataFrame
        df_copy = self.df.copy()
        df_copy['cluster'] = pd.Series(final_cluster)
        clusteredResult = df_copy.groupby('cluster')
        
        isCapacityExceedByAnyCluster = False
        cluster_stats = []
        
        for cluster_label, group in clusteredResult:
            # Skip noise points (cluster label -1)
            if cluster_label == -1:
                print(f"Noise points: {len(group)} customers (will need individual handling)")
                continue
                
            total_demand = group['DEMAND'].sum()
            cluster_size = len(group)
            isCapacityExceeded = total_demand > self.vehicleCapacity
            
            if isCapacityExceeded:
                isCapacityExceedByAnyCluster = True
                
            status = "Vehicle Capacity Exceeded" if isCapacityExceeded else "OK"
            print(f"Cluster {cluster_label} - Size: {cluster_size}, Demand: {total_demand}, Status: {status}")
            
            cluster_stats.append({
                'cluster_label': cluster_label,
                'size': cluster_size,
                'demand': total_demand,
                'capacity_exceeded': isCapacityExceeded
            })
        
        # Handle capacity constraint violations
        if isCapacityExceedByAnyCluster:
            print("\nCapacity constraints violated!")
            exceeded_clusters = [stat for stat in cluster_stats if stat['capacity_exceeded']]
            print(f"Clusters exceeding capacity: {[c['cluster_label'] for c in exceeded_clusters]}")
            print("Consider:")
            print("- Reducing max_cluster_size_const")
            print("- Using the recursive DBSCAN approach")
            print("- Increasing vehicle capacity")
            return None
        
        print(f"\nAll capacity constraints satisfied! Proceeding with TSP optimization...")
        return self.applyTSP(df_copy)  # Use the copy with cluster labels
    

###########################################~~~~~Recursive DBSCAN~~~~~#################################################
    def recursiveDBScan(self, min_radius: float, max_radius: float, min_no_clusters: int, df: pd.DataFrame):
        print(f"Radius range: [{min_radius:.3f}, {max_radius:.3f}], min_clusters: {min_no_clusters}")
        
        best_clusters = None
        min_radius_const = min_radius
        orders = df.iloc[:, [1, 2]].to_numpy() 
        numberOfOrders = len(orders)
        best_average_cluster_size = 0
        best_radius = 0
        
        # Store original bounds
        original_min = min_radius
        original_max = max_radius
        
        # Binary search for optimal radius
        while min_radius < max_radius:
            radius = (min_radius + max_radius) * 0.5
            clusters = self.hp.DBScan(orders, radius, min_samples=5)
            
            # Count actual clusters (excluding noise points labeled -1)
            unique_clusters = np.unique(clusters)
            actual_clusters = unique_clusters[unique_clusters != -1]  # Remove noise
            noOfClusters = len(actual_clusters)
            
            print(f"\nRadius {radius:.3f}: {noOfClusters} clusters, {np.sum(clusters == -1)} noise points")
            
            if noOfClusters < min_no_clusters:
                max_radius = radius-1
                print(f"reducing max_radius to {max_radius:.3f}")
            else:
                min_radius = radius+1
                average_cluster_size = numberOfOrders / noOfClusters if noOfClusters > 0 else 0
                if average_cluster_size > best_average_cluster_size:
                    best_average_cluster_size = average_cluster_size
                    best_clusters = clusters.copy()
                    best_radius = radius
                    print(f"\nNew best: {noOfClusters} clusters, avg size {average_cluster_size:.1f}")
            
            # Convergence check
            if abs(max_radius - min_radius) < 0.01:
                break
        
        if best_clusters is None: 
            print(f"No solution found in range [{original_min:.3f}, {original_max:.3f}]")
            # Return original dataframe with single cluster label if no clustering works
            df_result = df.copy()
            df_result['cluster'] = 0  # Assign all points to cluster 0
            return df_result
        
        print(f"Best clustering: radius {best_radius:.3f}, {len(np.unique(best_clusters[best_clusters != -1]))} clusters")
        
        # Create working dataframe with cluster assignments
        df_working = df.copy()
        df_working['cluster'] = best_clusters
        
        # Process each cluster
        final_clusters = []
        next_cluster_id = 0  # Start fresh cluster numbering
        
        # Handle noise points first (cluster == -1)
        noise_points = df_working[df_working['cluster'] == -1].copy()
        if len(noise_points) > 0:
            print(f"\nProcessing {len(noise_points)} noise points as individual clusters")
            for _, point in noise_points.iterrows():
                point_df = point.to_frame().T
                point_df['cluster'] = next_cluster_id
                final_clusters.append(point_df)
                next_cluster_id += 1
        
        # Process actual clusters
        clustered_data = df_working[df_working['cluster'] != -1]
        if len(clustered_data) > 0:
            clusteredResult = clustered_data.groupby('cluster')
            
            for cluster_label, cluster in clusteredResult:
                cluster_demand = cluster['DEMAND'].sum()
                cluster_size = len(cluster)
                print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"Cluster {cluster_label}: {cluster_size} points, demand {cluster_demand}")
                if cluster_demand > self.vehicleCapacity:
                    print(f"\nCapacity exceeded ({cluster_demand} > {self.vehicleCapacity})")
                    print("################...recursing...########################")
                    # Prepare data for recursion (remove cluster column)
                    cluster_for_recursion = cluster.drop('cluster', axis=1, errors='ignore')
                    
                    # Recursive call with adjusted parameters
                    sub_result = self.recursiveDBScan(
                        min_radius=min_radius_const * 0.8,
                        max_radius=best_radius * 0.9,  # Slightly smaller than current radius
                        min_no_clusters=2,
                        df=cluster_for_recursion
                    )
                    
                    if sub_result is not None and len(sub_result) > 0:
                        # Relabel clusters to avoid conflicts
                        sub_result = sub_result.copy()
                        sub_result['cluster'] = sub_result['cluster'] + next_cluster_id
                        next_cluster_id = sub_result['cluster'].max() + 1
                        final_clusters.append(sub_result)
                        print(f"\nRecursion successful: split into {len(sub_result['cluster'].unique())} subclusters")
                        
                    else:
                        print(f"\nRecursion failed, keeping single overloaded cluster")
                        cluster_copy = cluster.copy()
                        cluster_copy['cluster'] = next_cluster_id
                        final_clusters.append(cluster_copy)
                        next_cluster_id += 1
                    print("#############################################################")
                else:
                    # Cluster satisfies capacity constraint
                    print(f"\nCapacity OK, keeping cluster")
                    cluster_copy = cluster.copy()
                    cluster_copy['cluster'] = next_cluster_id
                    final_clusters.append(cluster_copy)
                    next_cluster_id += 1
        
        # Combine all processed clusters
        if final_clusters:
            final_result = pd.concat(final_clusters, ignore_index=True)
            print(f"\nFinal result: {len(final_result)} points in {len(final_result['cluster'].unique())} clusters")
        else:
            print(f"\nNo clusters processed, returning original data")
            final_result = df.copy()
            final_result['cluster'] = 0
        
        # Final capacity check and reporting
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"\n--- Final Capacity Check ---")
        isCapacityExceedByAnyCluster = False
        
        for cluster_id in final_result['cluster'].unique():
            cluster_data = final_result[final_result['cluster'] == cluster_id]
            total_demand = cluster_data['DEMAND'].sum()
            isCapacityExceeded = total_demand > self.vehicleCapacity
            if isCapacityExceeded:
                isCapacityExceedByAnyCluster = True
            status = "EXCEEDED" if isCapacityExceeded else "OK"
            print(f"Cluster {cluster_id}: {len(cluster_data)} points, demand {total_demand}, status {status}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if isCapacityExceedByAnyCluster:
            print("\nSome clusters still exceed capacity constraints")
        else:
            print("\nAll capacity constraints satisfied")
        
        return final_result
    
###########################################~~~~~Recursive DBSCAN SOLVER~~~~~#################################################
    def applyRecursiveDBScan(self,min_radius:float,max_radius:float,min_no_clusters:int):
        cluster =self.recursiveDBScan(min_radius=min_radius,max_radius=max_radius,min_no_clusters=min_no_clusters,df=self.df)
        if cluster is not None:
            return self.applyTSP(clusters=cluster)

###########################################~~~~~TODO Call DIFFERENT SOlver~~~~~#################################################

    def applyTSP(self,clusters):
        print("\n~~~~~~~~~~~Applying TSP~~~~~~~~~~~")
        return clusters