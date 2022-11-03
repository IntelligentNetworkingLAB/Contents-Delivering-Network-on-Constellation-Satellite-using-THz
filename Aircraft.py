from cmath import sqrt


class Aircraft:
    number = 0
    def __init__(self, x=0, y=0, altitude=0, source=(0, 0, 0), destination=(1, 1, 1)):
        '''
        Initiate the position
        Unit is km
        '''
        self.acNumber = Aircraft.number
        Aircraft.number = Aircraft.number+1
        self.x = x
        self.y = y
        self.altitude = altitude
        self.source = source
        self.destination = destination

    def getDestination(self):
        print(self.acNumber, "'s destination is : ", self.destination)
        return self.destination    
    
    def getVelocity(self, mode="even", totaltime=10):
        distance = 0
        
        for i in range(3):
            distance = distance + (self.destination[i] - self.source[i])**2
            
        velocity = sqrt(distance) / totaltime

        print(self.acNumber, "'s velocity is : ", velocity)
        return velocity

    

if __name__ == "__main__":
    ac1 = Aircraft(43, 22, 1000, (43, 22, 1000), (199, 200, 1000))
    dst = ac1.getDestination()
    vs1 = ac1.getVelocity()
    ac2 = Aircraft(43, 22, 1000, (43, 22, 1000), (199, 2100, 1000))
    dst = ac2.getDestination()
    vs2 = ac2.getVelocity()

    

    