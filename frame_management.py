
class Box_List():
    """
    Class to keep last n boxes
    Circular List implementation
    
    """
    def __init__(self):
        self.box_list = []
        self.index = 0
        self.full_list = False
        self.count = 8
        self.not_found = 0
        
    def add_box(self, bx):
        """
        Adds a Line to the list
    
        """
        if self.full_list == False:
            self.box_list.append(bx)
            self.index = (self.index + 1) 
            if self.index >= self.count:
                self.full_list = True
                self.index = self.index  % self.count 
            
        else:
            self.box_list[self.index] = bx
            self.index = (self.index + 1) % self.count
    
            
    def get_box_data(self):
        """
        Calculates the Line data given all line data in list
    
        """
        new_box = Box()
        
        for bx in self.box_list :
            new_box.top_x = new_box.top_x + bx.top_x
            new_box.top_y = new_box.top_y + bx.top_y
            new_box.bottom_x = new_box.bottom_x + bx.bottom_x
            new_box.bottom_y = new_box.bottom_y + bx.bottom_y
            centerx = (bx.top_x + bx.bottom_x) // 2
            centery = (bx.top_y + bx.bottom_y) // 2
            new_box.center_x = new_box.center_x + centerx
            new_box.center_y = new_box.center_y + centery
            
        if self.full_list == True:
            new_box.top_x = new_box.top_x // self.count
            new_box.top_y = new_box.top_y // self.count
            new_box.bottom_x = new_box.bottom_x // self.count
            new_box.bottom_y = new_box.bottom_y // self.count
            new_box.center_x = new_box.center_x // self.count
            new_box.center_y = new_box.center_y // self.count
        else:
            new_box.top_x = new_box.top_x // self.index
            new_box.top_y = new_box.top_y // self.index
            new_box.bottom_x = new_box.bottom_x // self.index
            new_box.bottom_y = new_box.bottom_y // self.index
            new_box.center_x = new_box.center_x // self.index
            new_box.center_y = new_box.center_y // self.index
            
        return new_box
        
    def get_count(self):
        """
        return number of items in list
    
        """
        if self.full_list == True:
            return self.count
        else:
            return self.index
    
    def increment_not_found(self):
        self.not_found += 1
    
    def get_count_not_found(self):
        return self.not_found
        
    
               
class Box():
    """
    Class to hold characteristics for box 
    
    """
    def __init__(self):
        self.top_x = 0
        self.top_y = 0
        self.bottom_x = 0
        self.bottom_y = 0
        self.center_x = 0
        self.center_y = 0
