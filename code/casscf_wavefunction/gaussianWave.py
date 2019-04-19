import numpy as np

class gaussianWavefunction:

    def __init__(self,fchkFileName,logFileName):
        f = open(fchkFileName)
        for line in f:
            if "Number of atoms" in line:
                self.nAtoms = int(line.split("I")[1])
            elif "Number of basis functions" in line:
                self.nBasisFunctions = int(line.split("I")[1])
            elif "Number of contracted shells" in line:
                self.nContractedShells = int(line.split("I")[1])
            elif "Current cartesian coordinates" in line:
                N = int(line.split("N=")[1])
                nLines = N//5
                if (N%5 > 0): nLines+=1
                atomPos = []
                for i in range(nLines):
                    for temp in f.readline().split():
                        atomPos.append(float(temp))
                self.atomPos = np.array(atomPos).reshape((self.nAtoms,3))
            elif "Shell types" in line:
                N = int(line.split("N=")[1])
                nLines = N//6
                if (N%6 > 0): nLines+=1
                shellType = []
                for i in range(nLines):
                    for temp in f.readline().split():
                        shellType.append(int(temp))
                self.shellType = np.array(shellType)
            elif "Number of primitives per shell" in line:
                N = int(line.split("N=")[1])
                nLines = N//6
                if (N%6 > 0): nLines+=1
                primitivesPerShell = []
                for i in range(nLines):
                    for temp in f.readline().split():
                        primitivesPerShell.append(int(temp))
                self.primitivesPerShell = np.array(primitivesPerShell)
            elif "Shell to atom map" in line:
                N = int(line.split("N=")[1])
                nLines = N//6
                if (N%6 > 0): nLines+=1
                shellAtom = []
                for i in range(nLines):
                    for temp in f.readline().split():
                        shellAtom.append(int(temp))
                self.shellAtom = np.array(shellAtom)
            elif "Primitive exponents" in line:
                N = int(line.split("N=")[1])
                nLines = N//5
                if (N%5 > 0): nLines+=1
                primitiveExponents = []
                for i in range(nLines):
                    for temp in f.readline().split():
                        primitiveExponents.append(float(temp))
                self.primitiveExponents = np.array(primitiveExponents)
            # note that P(S=P) check must come before Contraction coefficients check
            elif "P(S=P) Contraction coefficients" in line:
                N = int(line.split("N=")[1])
                nLines = N//5
                if (N%5 > 0): nLines+=1
                pspContractionCoefficients = []
                for i in range(nLines):
                    for temp in f.readline().split():
                        pspContractionCoefficients.append(float(temp))
                self.pspContractionCoefficients = np.array(pspContractionCoefficients)
            elif "Contraction coefficients" in line:
                N = int(line.split("N=")[1])
                nLines = N//5
                if (N%5 > 0): nLines+=1
                contractionCoefficients = []
                for i in range(nLines):
                    for temp in f.readline().split():
                        contractionCoefficients.append(float(temp))
                self.contractionCoefficients = np.array(contractionCoefficients)
            elif "Coordinates of each shell" in line:
                N = int(line.split("N=")[1])
                nLines = N//5
                if (N%5 > 0): nLines+=1
                shellPos = []
                for i in range(nLines):
                    for temp in f.readline().split():
                        shellPos.append(float(temp))
                self.shellPos = np.array(shellPos).reshape((self.nContractedShells,3))
            elif "Alpha MO coefficients" in line:
                N = int(line.split("N=")[1])
                nLines = N//5
                if (N%5 > 0): nLines+=1
                MOCoeff = []
                for i in range(nLines):
                    for temp in f.readline().split():
                        MOCoeff.append(float(temp))
                self.MOCoeff = np.array(MOCoeff).reshape((self.nBasisFunctions,self.nBasisFunctions))
        f.close()

        eigenValues = []
        moCoeff = []
        log = open(logFileName,"r")
        count = 0
        for line in log:
            if "Eigenvalues -- " in line:
                for temp in line.split('--')[1].split():
                    eigenValues.append(float(temp))
                #nLines = self.nBasisFunctions
                #tempCoeff = []
                #for i in range(nLines):
                #    tempCoeff.append([])
                #    for temp in log.readline()[23:].split():
                #        tempCoeff[i].append(float(temp))
                #tempCoeff = np.array(tempCoeff)
                #if count == 0:
                #    self.moCoeff = np.copy(tempCoeff)
                #    print(tempCoeff.shape)
                #else:
                #    self.moCoeff = np.column_stack((self.moCoeff,tempCoeff))
                #count += 1
        #self.moCoeff = np.array(moCoeff[-self.nBasisFunctions**2:]).reshape((self.nBasisFunctions,self.nBasisFunctions))

        eigenValues = np.array(eigenValues)
        eigenValues = np.sqrt(eigenValues/2.0)
        self.occupancies = eigenValues[-self.nBasisFunctions:]
        log.close()


    def mo_values(self,r):

        # determine value of each basis function at r
        basisRValue = []
        primitiveCount = 0
        functionCount = 0
        for shell, shellType in enumerate(self.shellType):
            # determine number of basis functions per shell type
            if shellType == 0:
                functionsPerShell = 1 # s-type
                tempVal = self.stype(self.primitivesPerShell[shell],self.contractionCoefficients[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.primitiveExponents[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.shellPos[shell,:],r)
                basisRValue.append(tempVal)
                primitiveCount += self.primitivesPerShell[shell]
            elif shellType == -1:
                functionsPerShell = 4 # sp-type
                # compute s-type with these exponents and contraction coefficients
                basisRValue.append(self.stype(self.primitivesPerShell[shell],self.contractionCoefficients[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.primitiveExponents[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.shellPos[shell,:],r))
                ptemp  = self.ptype(self.primitivesPerShell[shell],self.pspContractionCoefficients[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.primitiveExponents[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.shellPos[shell,:],r)
                for val in ptemp:
                    basisRValue.append(val)
                primitiveCount += self.primitivesPerShell[shell]
            elif shellType == 1:
                functionsPerShell = 3 # p-type
                ptemp  = self.ptype(self.primitivesPerShell[shell],self.contractionCoefficients[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.primitiveExponents[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.shellPos[shell,:],r)
                for val in ptemp:
                    basisRValue.append(val)
                primitiveCount += self.primitivesPerShell[shell]
            elif shellType == -2:
                functionsPerShell = 5 # 5d-type
                dtemp  = self.d5type(self.primitivesPerShell[shell],self.contractionCoefficients[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.primitiveExponents[primitiveCount:primitiveCount+self.primitivesPerShell[shell]],self.shellPos[shell,:],r)
                for val in dtemp:
                    basisRValue.append(val)
                primitiveCount += self.primitivesPerShell[shell]
            #elif shellType == 2:
                #functionsPerShell = 6 # 6d-type
            functionCount += functionsPerShell
        # convert to numpy array
        basisRValue = np.array(basisRValue)
        # return value
        #return np.dot(self.occupancies,np.dot(basisRValue,self.MOCoeff.T))
        return np.dot(basisRValue,self.MOCoeff.T)


    # compute value of s-type gaussian basis functions at position r
    def stype(self,n,c,zeta,r0,r):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # r0 is position (x,y,z) of basis function
        # r is position (x,y,z) to evaluate basis function
        # get displacement vector
        diff = r-r0
        r2 = np.dot(diff,diff)
        # compute normalization constant
        norm = 0.0
        for i in range(n):
            for j in range(n):
                norm += c[i]*c[j]/(zeta[i]+zeta[j])**1.5
        norm = np.pi**-0.75 * norm**-0.5
        # sum over primitives (if any)
        psiR = 0.0
        for i in range(n):
            psiR += c[i] * np.exp(-zeta[i]*r2)
        return psiR*norm

    # compute value of p-type gaussian basis functions at position r
    def ptype(self,n,c,zeta,r0,r):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # r0 is position (x,y,z) of basis function
        # r is position (x,y,z) to evaluate basis function
        # get displacement vector
        diff = r-r0
        r2 = np.dot(diff,diff)
        # compute normalization constant
        norm = 0.0
        for i in range(n):
            for j in range(n):
                norm += c[i]*c[j]/(zeta[i]+zeta[j])**2.5
        norm = np.sqrt(2.0)*np.pi**-0.75 * norm**-0.5
        # compute value of p-type at this position
        psiTemp = 0.0
        for i in range(n):
            psiTemp += c[i] * np.exp(-zeta[i]*r2)
        # multiply by x, y, or z to get p-type function
        psiR = psiTemp * diff
        # return array of psi values
        #return np.zeros(3,dtype=float)
        return psiR*norm

    # compute value of 5d-type gaussian basis functions at position r
    def d5type(self,n,c,zeta,r0,r):
        # n is number of primitives in CGTO
        # c is array of contraction coefficients (size n)
        # zeta is array of exponents (size n)
        # r0 is position (x,y,z) of basis function
        # r is position (x,y,z) to evaluate basis function
        # compute normalization constant
        norm = np.empty(5,dtype=float)
        norm[0] = 2/np.sqrt(3.0)*(2.0/np.pi)**0.75*zeta[0]**1.75
        norm[1] = 2*(2.0/np.pi)**0.75*zeta[0]**1.75
        norm[2] = norm[3] = norm[4] = 4*(2.0/np.pi)**0.75*zeta[0]**1.75
        # get displacement vector
        diff = r-r0
        r2 = np.dot(diff,diff)
        # compute value of d-type at this position
        psiTemp = c[0] * np.exp(-zeta[0]*r2)
        # values of d-type multipliers
        d5 = np.array([2*diff[2]**2-diff[0]**2-diff[1]**2,diff[0]**2-diff[1]**2,diff[0]*diff[1],diff[0]*diff[2],diff[1]*diff[2]])
        psiR = d5 * psiTemp
        for i in range(5):
            psiR[i] *= norm[i]
        return psiR
