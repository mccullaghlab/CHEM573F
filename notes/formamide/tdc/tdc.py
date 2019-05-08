import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm

c = 2.99792458e10 # cm-sec-1
energyConvert = 6.02214E46   # convert from J/amu/A^2 to 1/sec^2
convert = 1E30*(3.1625E-25)**2
dipoleUnitConvert = 1.0/np.sqrt(42.2561)
fwhm_to_sigma = 1.0/(8.0*np.sqrt(np.log(2.0)))
gauss_const = 1.0/np.sqrt(2*np.pi)

def gaussian(x,fwhm,x0):
    sigma = fwhm_to_sigma * fwhm
    return gauss_const/sigma*np.exp(-(x-x0)**2/(2*sigma))

def lorentzian(x,fwhm,x0):
    return 0.5*fwhm/(np.pi*((x-x0)**2+0.25*fwhm**2))

class freq:

    dipoleUnitConvert = 1.0/np.sqrt(42.2561)
    forceConstantUnitConvert = 1E-18

    def __init__(self,logFileName):
        log = open(logFileName,"r")
        xyz = []
        self.frequencies = []
        self.ramanIntensities = []
        self.irIntensities = []
        self.dipoleDerivative = []
        self.dipoleMoment = []
        self.reducedMasses = []
        self.forceConstant = []
        self.dipoleDerivativeCount = 0
        normalModeCount = 0
        for line in log:
            if "NAtoms=" in line:
                temp = line.split()
                self.nAtoms = int(temp[1])
            elif "Center     Atomic      Atomic             Coordinates (Angstroms)" in line:
                log.readline()
                log.readline()
                readCoor = "pass"
                self.atomPositions = []
                atom = 0
                while readCoor == "pass":
                    temp = log.readline()
                    if "-------------------------------------------------------------------" in temp:
                        readCoor = "done"
                    else:
                        temp = temp.split()
                        self.atomPositions.append([])
                        for k in range(3):
                            self.atomPositions[atom].append(float(temp[k+3]))
                        atom += 1
                self.atomPositions = np.array(self.atomPositions)
            elif "Dipole derivatives wrt mode" in line:
                temp = line.split(":")[1].split()
                self.dipoleDerivative.append([])
                for i in range(len(temp)):
                    self.dipoleDerivative[self.dipoleDerivativeCount].append(float(temp[i].replace('D','E')))
                self.dipoleDerivativeCount += 1
            elif " Dipole moment (field-independent basis, Debye):" in line:
                temp = log.readline().split()
                for i in range(3):
                    self.dipoleMoment.append(float(temp[i*2+1]))
            elif "Frequencies -- " in line:
                for temp in line.split('--')[1].split():
                    self.frequencies.append(float(temp))
            elif "Red. masses -- " in line:
                for temp in line.split('--')[1].split():
                    self.reducedMasses.append(float(temp))
            elif "Frc consts" in line:
                for temp in line.split('--')[1].split():
                    self.forceConstant.append(float(temp))
            elif "IR Inten    --" in line:
                for temp in line.split('--')[1].split():
                    self.irIntensities.append(float(temp))
            elif "Raman Activ --" in line:
                for temp in line.split('--')[1].split():
                    self.ramanIntensities.append(float(temp))
            elif "Atom  AN      X      Y      Z" in line:
                tempXyz = []
                for atom in range(self.nAtoms):
                    temp = log.readline().split()
                    tempXyz.append([])
                    for i in range(2,len(temp)):
                        tempXyz[atom].append(float(temp[i]))
                tempXyz = np.array(tempXyz)
                if normalModeCount == 0:
                    self.normalModes = np.copy(tempXyz)
                else:
                    self.normalModes = np.column_stack((self.normalModes,tempXyz))
                normalModeCount += 1
        log.close()
        self.nModes = len(self.frequencies)
        self.irIntensities = np.array(self.irIntensities)
        self.frequencies = np.array(self.frequencies)

        self.reducedMasses = np.array(self.reducedMasses)
        self.forceConstant = np.array(self.forceConstant)*self.forceConstantUnitConvert
        vals, self.molecularBasis = np.linalg.eigh(np.dot(self.atomPositions.T,self.atomPositions))
        if len(self.ramanIntensities) > 0:
            self.ramanIntensities = np.array(self.ramanIntensities)
        if self.dipoleDerivativeCount > 0:
            self.dipoleDerivative = np.array(self.dipoleDerivative)*self.dipoleUnitConvert
            self.dipoleDerivativeMolecularBasis = np.dot(self.dipoleDerivative,self.molecularBasis).T
            self.oscStrength = np.empty(self.nModes,dtype=float)
            for i in range(self.nModes):
                self.oscStrength[i] = np.linalg.norm(self.dipoleDerivative[i,:])

    def select_modes(self,modes):
        if self.dipoleDerivativeCount > 0:
            self.dipoleDerivative = self.dipoleDerivative[modes,:]
            self.dipoleDerivativeMolecularBasis = self.dipoleDerivativeMolecularBasis[:,modes]
            self.oscStrength = self.oscStrength[modes]
        if len(self.ramanIntensities) > 0:
                    self.ramanIntensities = self.ramanIntensities[modes]
        self.nModes = len(modes)
        self.forceConstant = self.forceConstant[modes]
        self.irIntensities = self.irIntensities[modes]
        self.reducedMasses = self.reducedMasses[modes]
        self.frequencies = self.frequencies[modes]

    def sticks(self,waveNumberBuffer=50,deltaWaveNumber=0.01):
        waveMin = np.amin(self.frequencies)-waveNumberBuffer
        waveMax = np.amax(self.frequencies)+waveNumberBuffer
        x = np.arange(waveMin,waveMax,deltaWaveNumber)
        y = np.zeros(x.shape,dtype=float)
        for i in range(self.nModes):
            xbin = int((self.frequencies[i] - waveMin) / deltaWaveNumber)
            y[xbin] = self.irIntensities[i]
        return np.column_stack((x,y))
    
    def lorentzian_convolution(self,fwhm=10.0,waveNumberBuffer=50,deltaWaveNumber=0.01):
        waveMin = np.amin(self.frequencies)-waveNumberBuffer
        waveMax = np.amax(self.frequencies)+waveNumberBuffer
        x = np.arange(waveMin,waveMax,deltaWaveNumber)
        y = np.zeros(x.shape,dtype=float)
        for i in range(self.nModes):
            y += self.irIntensities[i]*lorentzian(x,fwhm,self.frequencies[i])
        return np.column_stack((x,y))
    
    def gaussian_convolution(self,fwhm=10.0,waveNumberBuffer=50,deltaWaveNumber=0.01):
        waveMin = np.amin(self.frequencies)-waveNumberBuffer
        waveMax = np.amax(self.frequencies)+waveNumberBuffer
        x = np.arange(waveMin,waveMax,deltaWaveNumber)
        y = np.zeros(x.shape,dtype=float)
        for i in range(self.nModes):
            y += self.irIntensities[i]*guassian(x,fwhm,self.frequencies[i])
        return np.column_stack((x,y))
        
class tdc:

    convert = 1E30*(3.1625E-25)**2

    def __init__ (self,monomerFreq,multimerPos):

        self.nAtoms = multimerPos.shape[0]
        self.nMols = self.nAtoms // monomerFreq.nAtoms
        self.nModes = monomerFreq.nModes * self.nMols

        # compute H0 - diagonal Hessian matrix based on molecular normal modes
        self.H0 = np.zeros((self.nModes,self.nModes),dtype=float)
        count = 0
        for i in range(self.nMols):
            for j in range(monomerFreq.nModes):
                self.H0[count,count] = monomerFreq.forceConstant[j]/monomerFreq.reducedMasses[j]
                count += 1

        # compute V - matrix of couplings between each molecule based on TDC
        self.V = np.zeros(self.H0.shape,dtype=float)
        count1 = 0
        for mol1 in range(self.nMols-1):
            meanCenteredCoord = np.copy(multimerPos[mol1*monomerFreq.nAtoms:(mol1+1)*monomerFreq.nAtoms,:])
            mol1Mean = np.mean(meanCenteredCoord,axis=0)
            meanCenteredCoord -= mol1Mean
            vals, mol1MolecularBasis = np.linalg.eigh( np.dot(meanCenteredCoord.T, meanCenteredCoord) )
            for mode1 in range(monomerFreq.nModes):
                u1 = np.dot(mol1MolecularBasis,monomerFreq.dipoleDerivativeMolecularBasis[:,mode1])
                for mol2 in range(mol1+1,self.nMols):
                    meanCenteredCoord = np.copy(multimerPos[mol2*monomerFreq.nAtoms:(mol2+1)*monomerFreq.nAtoms,:])
                    mol2Mean = np.mean(meanCenteredCoord,axis=0)
                    meanCenteredCoord -= mol2Mean
                    vals, mol2MolecularBasis = np.linalg.eigh( np.dot(meanCenteredCoord.T, meanCenteredCoord) )
                    for mode2 in range(monomerFreq.nModes):
                        u2 = np.dot(mol2MolecularBasis,monomerFreq.dipoleDerivativeMolecularBasis[:,mode2])
                        r = mol2Mean - mol1Mean
                        self.V[mol1*monomerFreq.nModes+mode1, mol2*monomerFreq.nModes + mode2] = self.tdc_coupling(u1,u2,r)
                        # symmetrize
                        self.V[mol2*monomerFreq.nModes+mode2, mol1*monomerFreq.nModes + mode1] = self.V[mol1*monomerFreq.nModes+mode1, mol2*monomerFreq.nModes + mode2] 


        # generate total Hessian and diagonalize
        self.H1 = self.H0 + self.V
        vals, self.vecs = np.linalg.eigh(self.H1)
        # save new frequencies
        self.frequencies = 1.0/(2.0*np.pi*c) * np.sqrt(vals*energyConvert)
        # save new IR intensities
        us = np.empty((self.nModes,3),dtype=float)
        for mol1 in range(self.nMols):
            meanCenteredCoord = np.copy(multimerPos[mol1*monomerFreq.nAtoms:(mol1+1)*monomerFreq.nAtoms,:])
            mol1Mean = np.mean(meanCenteredCoord,axis=0)
            meanCenteredCoord -= mol1Mean
            vals, mol1MolecularBasis = np.linalg.eigh( np.dot(meanCenteredCoord.T, meanCenteredCoord) )
            for mode in range(monomerFreq.nModes):
                us[mol1*monomerFreq.nModes + mode] = np.dot(mol1MolecularBasis,monomerFreq.dipoleDerivativeMolecularBasis[:,mode])/dipoleUnitConvert
        self.irIntensities = np.empty(self.nModes,dtype=float)
        for mode in range(self.nModes):
            u = np.zeros(3,dtype=float)
            for mode2 in range(self.nModes):
                u += self.vecs[mode2,mode]*us[mode2]
            self.irIntensities[mode] = np.dot(u,u)

    def tdc_coupling(self,u1,u2,r):
        rMag = np.linalg.norm(r)
        return (np.dot(u1,u2)/(rMag**3) - 3.0*(np.dot(u1,r)*np.dot(u2,r))/(rMag**5))*convert

    def sticks(self,waveNumberBuffer=50,deltaWaveNumber=0.01):
        waveMin = np.amin(self.frequencies)-waveNumberBuffer
        waveMax = np.amax(self.frequencies)+waveNumberBuffer
        x = np.arange(waveMin,waveMax,deltaWaveNumber)
        y = np.zeros(x.shape,dtype=float)
        for i in range(self.nModes):
            xbin = int((self.frequencies[i] - waveMin) / deltaWaveNumber)
            y[xbin] = self.irIntensities[i]
        return np.column_stack((x,y))
    
    def lorentzian_convolution(self,fwhm=10.0,waveNumberBuffer=50,deltaWaveNumber=0.01):
        waveMin = np.amin(self.frequencies)-waveNumberBuffer
        waveMax = np.amax(self.frequencies)+waveNumberBuffer
        x = np.arange(waveMin,waveMax,deltaWaveNumber)
        y = np.zeros(x.shape,dtype=float)
        for i in range(self.nModes):
            y += self.irIntensities[i]*lorentzian(x,fwhm,self.frequencies[i])
        return np.column_stack((x,y))
    
    def gaussian_convolution(self,fwhm=10.0,waveNumberBuffer=50,deltaWaveNumber=0.01):
        waveMin = np.amin(self.frequencies)-waveNumberBuffer
        waveMax = np.amax(self.frequencies)+waveNumberBuffer
        x = np.arange(waveMin,waveMax,deltaWaveNumber)
        y = np.zeros(x.shape,dtype=float)
        for i in range(self.nModes):
            y += self.irIntensities[i]*guassian(x,fwhm,self.frequencies[i])
        return np.column_stack((x,y))
    
    def plot_sticks(self,waveNumberBuffer=50,deltaWaveNumber=0.01):
        # setup plot parameters
        fig = plt.figure(figsize=(10,8), dpi= 80, facecolor='w', edgecolor='k')
        ax = plt.subplot(111)
        ax.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        ax.set_xlabel("Frequencies (cm$^{-1}$)",size=20)
        ax.set_ylabel("IR Intensities (km/mol)",size=20)
        plt.tick_params(axis='both',labelsize=20)
        waveMin = np.amin(self.frequencies)-waveNumberBuffer
        waveMax = np.amax(self.frequencies)+waveNumberBuffer
        x = np.arange(waveMin,waveMax,deltaWaveNumber)
        y = np.zeros(x.shape,dtype=float)
        for i in range(self.nModes):
            xbin = int((self.frequencies[i] - waveMin) / deltaWaveNumber)
            y[xbin] = self.irIntensities[i]
        ax.plot(x,y)



