import pickle
f = open('test.pkl','wb')
x = pickle.load(f)
f.close()
#the X variable is a list of dictionaries where list index is frame number. The dictionary contents are same
#This Should suffice as basis for your start. I have already uploaded the pickle file. It should contain 329 dictionaries
