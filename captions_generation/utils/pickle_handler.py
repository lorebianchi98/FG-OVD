import pickle

def saveObject(obj, path):
    """"Save an object using the pickle library on a file
    
    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + path + '.pkl')
    with open(path + ".pkl", 'wb') as fid:
        pickle.dump(obj, fid)
        
def loadObject(path):
    """"Load an object from a file
    
    :param fileName: str. Name of the file of the object to load
    :return: obj: undefined. Object loaded
    """
    try:
        with open(path + '.pkl', 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None   