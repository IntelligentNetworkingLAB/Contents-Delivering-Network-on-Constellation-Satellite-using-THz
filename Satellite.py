from Aircraft import Aircraft

class Satellite(Aircraft):
    
    def __init__(self, x=0, y=0, z=0, source=(0, 0, 0), destination=(1, 1, 1), storage = 0, bandwidth = 0):
        super().__init__(x, y, z, source, destination)
        self.storage = storage
        self.bandwidth = bandwidth
    
    def getStorage(self):
        print(self.storage)
        return self.storage

if __name__ == "__main__":
    st1 = Satellite(43, 22, 1000, (43, 22, 1000), (199, 200, 1000), 100, 200000)
    st1.getStorage()
    st1.getDestination()
    st1.getVelocity()
