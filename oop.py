# Примеры ООП в python

class classLister():
    def __init__(self):
        self.lister_prperty = 99


    def __str__(self):
        str = "<Instance of class {} in place {} with attributes {}>".format(self.__class__.__name__, id(self), self.__attrnames_list_by_dir())
        return  str

    def __attrnames_list(self):
        result = ""
        for attr_name, attr_value in sorted(self.__dict__.items()):
            result += "next property {} = {}; ".format(attr_name, attr_value)

        return result
    def __attrnames_list_by_dir(self):
        result = ""
        for attr in dir(self):
            if attr[:2] == "__" and attr[-2:] == "__":
                result += "{}=<> \t".format(attr)
            else:
                result += " {}={} \t".format(attr, getattr(self, attr))
        return result

class myclass(classLister):

    def __init__(self):
        self.prop1 = "Имя"
        self.prop2 = 10

    def changeProp1(self, new_prop1_value):
        self.prop1 = new_prop1_value

    def changeProp2(self, new_prop2_value):
        self.prop2 = new_prop2_value

    @staticmethod
    def static_method(word):
        print(word)

a = myclass()
print(a)


class Sensor:
    Instatnce = 0

    def __init__(self):
        Sensor.Instatnce += 1

    @staticmethod
    def getInstancesNumber():
        return Sensor.Instatnce

s1 = Sensor()
s2 = Sensor()
s3 = Sensor()

print(Sensor.getInstancesNumber(), s2.getInstancesNumber())