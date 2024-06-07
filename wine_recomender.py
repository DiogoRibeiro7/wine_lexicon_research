'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''


class WineRecommender:
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.columns_title = ["Rating", "Number of Ratings", "Price"]
        self.pivot_var = ["rating", "nbr_of_ratings", "price"]
    
    def get_pivot(self, column_name="rating"):
        """
        Args:
        column_name: The name of the column you want to pivot.
                     column_name ={"rating","nbr_of_ratings","price"}, default is "rating"
                
        Returns:
        A pivoted table (dataframe)
        """
        var_dict ={}
        for i in range(3):
            var_dict[self.pivot_var[i]] = pd.pivot_table(self.dataset, index=['Winery', 'Name'], values=[self.columns_title[i]], columns='Wine Type')
            var_dict[self.pivot_var[i]].columns=['Dessert', 'Fortified', 'Red', 'Rose՛', 'Sparkling', 'White']
            var_dict[self.pivot_var[i]] = var_dict[self.pivot_var[i]].fillna(0)
        
        if column_name == "nbr_of_ratings":
            return var_dict["nbr_of_ratings"]
        elif column_name == "price":
            return var_dict["price"]
        else:
            return var_dict["rating"]

    def get_indexes(self, suggestions, pivot_table):
        idx = 0
        sim = []
        wineries = []
        wine_names = []
        for i in suggestions:
            sim.append(pivot_table.index[i])
            wineries.append(sim[idx][0])
            wine_names.append(sim[idx][1])
            idx += 1

        indexes = []
        for index in range(len(wine_names)):
            ind = np.where((self.dataset['Winery'] == wineries[index]) & (self.dataset['Name'] == wine_names[index]))
            indexes.append(ind[0][0])
        
        return indexes
    
    def get_indices(self, suggestions, pivot_table):
        idx = 0
        sim = []
        wineries = []
        wine_names = []
        for i in suggestions[0]:
            sim.append(pivot_table.index[i])
            wineries.append(sim[idx][0])
            wine_names.append(sim[idx][1])
            idx += 1

        indices = []
        for index in range(len(wine_names)):
            ind = np.where((self.dataset['Winery'] == wineries[index]) & (self.dataset['Name'] == wine_names[index]))
            indices.append(ind[0][0])
        
        return indices

    def recommend(self, winery=None, name=None, pos=None, recommend_by="rating", method="KNeighbors", algorithm="brute", n_neighbors=11, sim_elem=11):
        """
        Args:
        winery: The name of the winery, if method = "Similarity"
        name: The name of the wine, if method = "Similarity"
        pos: The index position of the wine, if method= "KNeighbors"
        recommend_by: The method used for recommendation. must be a column in the dataframe.
        method: The algorithm used for recommendation. options={"KNeighbors","Similarity"}
        algorithm: The algorithm used in the recommendation method. options{"auto","ball_tree","brute"}. default is "brute"
        n_neighbors: The number of neighbors to use in the KNeighbors algorithm. default 10.
        sim_elem: The number of similar wines to return in the Similarity algorithm. default 10.
        
        Returns:
        A dataframe of recommended wines
        """
        
        if method == "KNeighbors":
            pivot_table = self.get_pivot(column_name=recommend_by) # getting the pivot_table of the dataframe
            pivot_table_sparse = csr_matrix(pivot_table) # converting the dense matrix as a sparse matrix
            rec_model=NearestNeighbors(algorithm=algorithm)
            rec_model.fit(pivot_table_sparse)
            
            distance, sugg = rec_model.kneighbors(pivot_table.iloc[pos,:].values.reshape(1,-1), n_neighbors= n_neighbors)
            indices=self.get_indices(sugg, pivot_table)
            
            print(f"Showing Recommendations for \n")
            
#           print the name of the wine your trying to recommend.
            print(self.dataset.loc[indices[0]])
            
            
           
            
#             returning the recommendations as  a dataframe
            return self.dataset.loc[indices[1:]]

        if method == "Similarity":
            df = self.dataset.copy()
#             features = df[self.columns_title]
            scaler = StandardScaler()
            features = scaler.fit_transform(df[self.columns_title])
            similarity = cosine_similarity(features)
            
#              find the index of the Wine you are recommending
            index = df[(df['Winery'].str.contains(winery))&(df['Name'].str.contains(name))].index[0]
            
            print("Showing Recommendations for \n")
            print(data.loc[index])
            
            wines_list = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x:x[1])[1:sim_elem]
            
            indexes = []
            for wine in wines_list:
                indexes.append(wine[0])
                
            return self.dataset.loc[indexes]
        