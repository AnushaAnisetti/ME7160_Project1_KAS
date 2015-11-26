__author__ = 'koorosh'
import matplotlib.pylab as plt
import numpy as np

# Define vertices coordiates
vertCoord = np.array([[-5,0],
                      [0,1],
                      [5, 0],
                      [0, -1]])
# number of node between each two vertices
N = 3
n = N + 2
nodeCoord = np.zeros([4*N + 4, 2])

nodeCoord[0, :] = vertCoord[0, :]
nodeCoordX = np.linspace(vertCoord[0, 0], vertCoord[1, 0], n)
nodeCoordY = (vertCoord[1, 1] - vertCoord[0, 1]) / \
             (vertCoord[1, 0] - vertCoord[0, 0]) * nodeCoordX + vertCoord[1, 1]
nodeCoord[1:N+1, 0] = nodeCoordX[1:-1]
nodeCoord[1:N+1, 1] = nodeCoordY[1:-1]

nodeCoord[N+1, :] = vertCoord[1, :]
nodeCoordX = np.linspace(vertCoord[1, 0], vertCoord[2, 0], n)
nodeCoordY = (vertCoord[2, 1] - vertCoord[1, 1]) / \
             (vertCoord[2, 0] - vertCoord[1, 0]) * nodeCoordX + vertCoord[1, 1]
nodeCoord[N+2:2*N+2, 0] = nodeCoordX[1:-1]
nodeCoord[N+2:2*N+2, 1] = nodeCoordY[1:-1]

nodeCoord[2*N+2, :] = vertCoord[2, :]
nodeCoordX = np.linspace(vertCoord[2, 0], vertCoord[3, 0], n)
nodeCoordY = (vertCoord[3, 1] - vertCoord[2, 1]) / \
             (vertCoord[3, 0] - vertCoord[2, 0]) * nodeCoordX + vertCoord[3, 1]
nodeCoord[2*N+3:3*N+3, 0] = nodeCoordX[1:-1]
nodeCoord[2*N+3:3*N+3, 1] = nodeCoordY[1:-1]

nodeCoord[3*N+3, :] = vertCoord[3, :]
nodeCoordX = np.linspace(vertCoord[3, 0], vertCoord[0, 0], n)
nodeCoordY = (vertCoord[0, 1] - vertCoord[3, 1]) / \
             (vertCoord[0, 0] - vertCoord[3, 0]) * nodeCoordX + vertCoord[3, 1]
nodeCoord[3*N+4:4*N+4, 0] = nodeCoordX[1:-1]
nodeCoord[3*N+4:4*N+4, 1] = nodeCoordY[1:-1]

np.savetxt('coord.txt', nodeCoord, '%2.2f')

# Define rotation in clockwise with respect to [0,0]
theta = 0
theta = theta * np.pi / 180
rMat = np.matrix([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])

for ni in range(0, nodeCoord.shape[0]):
    nodeCoord[ni, :] = (rMat * nodeCoord[ni, :].reshape([2, 1])).reshape([1, 2])

plt.figure()
plt.plot(nodeCoord[:, 0], nodeCoord[:, 1], 'k-')
plt.axis('equal')
plt.show()
