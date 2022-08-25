

class LinkedParameters: 

    '''
    represents parameters which are not bound to a single reaction object, 
    but are instead shared between multiple reactions. 
    '''

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return f'{self.name}'

    def __eq__(self, other):
        if isinstance(other, LinkedParameters):
            return self.name == other.name and self.value == other.value
        return False

    def __hash__(self):
        return hash((self.name, self.value))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, LinkedParameters):
            return self.name < other.name
        return False

    def __le__(self, other):
        if isinstance(other, LinkedParameters):
            return self.name <= other.name
        return False

    def __gt__(self, other):
        if isinstance(other, LinkedParameters):
            return self.name > other.name
        return False

    def __ge__(self, other):
        if isinstance(other, LinkedParameters):
            return self.name >= other.name
        return False
