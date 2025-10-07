from ucimlrepo import fetch_ucirepo 
def load_data():
    vertebral_column = fetch_ucirepo(id=212) 
    
    X = vertebral_column.data.features 
    y = vertebral_column.data.targets 

    X = X.values
    y = y.values

    return X,y
        