
class Direction:
    UP = 'UP'
    DOWN = 'DOWN'

class Status:
    UP = 'UP'
    DOWN = 'DOWN'
    IDLE = 'IDLE'

class Request:
    def __init__(self,l = 0):
        self.level = l
        
    def getLevel(self):
        return self.level

class ElevatorButton:
    def __init__(self,level,e):
        self.level = level
        self.elevator = e
        
    def pressButton(self):
        request = InternalRequest(self.level)
        self.elevator.handleInternalRequest(request);

class ExternalRequest(Request):
    def __init__(self,l = 0,d = None):
        Request.__init__(self,l)
        self.direction = d

    def getDirection(self):
        return self.direction

class InternalRequest(Request):
    def __init__(self,l = None):
        Request.__init__(self,l)

class Elevator:
    def __init__(self, n):
        # Keep them, don't modify.
        self.buttons = []
        self.upStops = []
        self.downStops = []
        for i in xrange(n):
            self.upStops.append(False)
            self.downStops.append(False)
        self.currLevel = 0
        self.status = Status.IDLE

    def insertButton(self,eb):
        self.buttons.append(eb)

    def handleExternalRequest(self,r):
    	# Write your code here  
 
        
    def handleInternalRequest(self,r):
		# Write your code here
        
        
    def openGate(self):
		# Write your code here

        
    def closeGate(self):
		# Write your code here  
        

    def noRequests(self, stops):
		for stop in stops:
		    if stop:
		        return False
		return True
	
    def elevatorStatusDescription(self):
        description = "Currently elevator status is : " + self.status + \
                      ".\nCurrent level is at: " + str(self.currLevel + 1) + \
                      ".\nup stop list looks like: " + self.toString(self.upStops) + \
                      ".\ndown stop list looks like:  " + self.toString(self.downStops) + \
                      ".\n*****************************************\n"
        return description
        
    @classmethod
    def toString(cls, stops):
        return str(stops).replace("False", "false").replace("True", "true")