#taken from http://stackoverflow.com/questions/7019283/automatically-type-cast-parameters-in-python
#This autoconverts values from string to the proper data type
#it has been modified to replace 'None' with None
def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("huh?")

def autoconvert(s):
    if(s == 'None'):
        return None
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s
