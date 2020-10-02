import numpy as np
import pandas as pd
import math
from IPython.display import display, Math, Latex
import fractions
import matplotlib.pyplot as plt

def calculate_V0(prices, quantity):
    """returns the V0 value hiven the price and the associated quantity"""
    return sum([p*q for (p, q) in zip(prices, quantity)])

def calculate_VT(prices, quantity):
    return sum([p*q for (p, q) in zip(prices, quantity)])

def discreteRate(rate, time, principal=1):
    return principal* ((r*time)+1)

def findUnknownRate(time, principal=1, finalValue=1):
    """solving the equation finalValue = principal* (1+ r*time)"""
    print("modified the equation Vt= V0*(1+r*t) to be ((VT-V0)/V0) / t = r")
    return ROI(principal, finalValue) / time

def ROI(V0, VT):
    return (VT-V0)/V0

def convertFrac(arr):
    return [fractions.Fraction(i) for i in arr]

def convertBack(arr):
    return [i.numerator/i.denominator for i in arr]

def elemProd(arr1, arr2, sumB=True):
    if not sumB:
        return [i*j for (i, j) in zip(arr1, arr2)]
    return sum([i*j for (i, j) in zip(arr1, arr2)])

def elemSq(arr):
    return [i*i for i in arr]

def convertListToDecimal(listA):
    return [frac if type(frac) == float else (frac if frac.denominator != 1 else frac.numerator) for frac in listA]

def checkForShortSelling(sigma1, sigma2, p12):
    if sigma1 >= sigma2:
        if p12 == 1:
            print("p12 = 1, there exist a deasable set with short selling s.t. sigmaV = 0.  each portfolio has the same sigmaV when sigma1 = sigma2")
        elif (sigma1/sigma2) < p12 < 1:
            print(f"feasable set with short selling, s.t. sigmaV < sigma1 {round(sigma1, 5)}, but without shortselling if sigmaV >= sigma1 {round(sigma1, 5)}")
        elif p12 == sigma1/sigma2:
            print("no short selling")
        elif -1 < p12 < sigma1/sigma2:
            print("feasable set without short selling, -1 < p12 < sigma1/sigma2")
        elif p12 == -1:
            print("there's a feasabale set without short selling, s.t. sigmaV < sigma1")
        else:
            print("wha?")
    else:
        checkForShortSelling(sigma2, sigma1, p12)

def makeCMatrix(correlationList, stdDevList):

    matrixSize = len(stdDevList)
    correlationArr = [[0 for _ in range(matrixSize)] for _ in range(matrixSize)]
    for i in range(matrixSize):
        for j in range(matrixSize):
            if i==j:
                correlationArr[i][j] = stdDevList[i]* stdDevList[i]
            else:
                correlationArr[i][j] = findCorrelation(i+1, j+1, correlationList) * (stdDevList[i] * stdDevList[j])

    cMatrix = np.array(correlationArr)
    print("\nC matrix is:")
    print(cMatrix)
    return cMatrix

def findCorrelation(i, j, correlation):
    for val in correlation:
        if val[0] == [i, j] or val[0] == [j, i]:
            return val[1]

def meanV(weights, means):
    return sum(w * m for (w, m) in zip(weights, means))

def find_wCwT(w, c):
    wC = np.matmul(w, c)
    print("intermediate calculation: w*C = ", wC)
    return np.matmul(wC, np.array(w).T)

def makeM(m, cInv, u):
    m, u = np.array(m), np.array(u)
    mCInv = np.matmul(m, cInv)
    uCInv = np.matmul(u, cInv)
    print("Intermediate Calculations:")
    print("M*C^-1 = ", mCInv)
    print("u*C^-1 = ", uCInv)

    mCInv_mT = np.matmul(mCInv, m.T)
    mCInv_uT = np.matmul(mCInv, u.T)
    uCInv_mT = np.matmul(uCInv, m.T)
    uCInv_uT = np.matmul(uCInv, u.T)

    M = [[mCInv_mT, uCInv_mT],
        [mCInv_uT, uCInv_uT]]

    print("\nhere is M:")
    print(np.array(M))
    return np.array(M)

def minWeights(u, cInv):
    min_weights = np.matmul(u, cInv)/ (np.matmul(np.matmul(u, cInv), np.array(u).T))
    print("\nThe weight in minimum variance portfolio is:")
    print(min_weights)
    return min_weights

def find_W_MVP(m, stdDevs, correlation):
    m, u = np.array(m), np.array([1 for _ in m])
    print("m is:", m, "\nu is:", u)
    cMatrix = np.array(makeCMatrix(correlation, stdDevs))
    cInverse = np.linalg.inv(cMatrix)
    print("\nhere is the inverse of C:\n", cInverse)
    print("\ncalculating M:")
    Mmatrix = makeM(m, cInverse, u)
    print("\nhere is M inverse:")
    MmatrixInv = np.linalg.inv(Mmatrix)
    print(MmatrixInv)
    minimumWeights = minWeights(u, cInverse)
    print("\nThe Expected return is:")
    Emin = elemProd(minimumWeights, m)
    print(Emin)
    print("\nVariance is:")
    wCwT = find_wCwT(minimumWeights, cMatrix)
    print("wCwT = Varinace = ")
    print(wCwT)
    stdDevMin = math.sqrt(wCwT)
    print("\nThe Std.Dev of the portfolio is:")
    print(stdDevMin)

def Variance_classic(points, probability):
    """inputs: points, probability
    """
    mean = meanV(points, probability)
    adjusted_points = [point-mean for point in points]
    adj_squared = [i*i for i in adjusted_points]
    variance = sum([a*b for (a, b) in zip(adj_squared, probability)])
    rhs = [str(p)+"*("+str(point)+"-"+str(mean)+")^2" for (p, point) in zip(probability, points)]
    rhs_joined = " + ".join(rhs)
    print("Var( ) = ", rhs_joined, "=", variance)
    return variance

def continuousCompounding(rate, time):
    return math.exp(rate*time)

def discreteCompounding(rate, time, periods=1):
    return (1+(rate/periods))**(time*periods)

def findContinuousRate(PV, FV, time):
    return math.log(PV/FV)/time

def logReturn(V0, VT):
    return math.log(VT/V0)

def findLogReturnRate(logReturn, time):
    print("replace V(0) with 1 amd V(T) with e^rt")
    return logReturn/time

def bond_start(start, time_to_maturity, coupon, FV, rate):
    coupons = sum([coupon*math.exp(-rate*i) for i in range(1, time_to_maturity+1-start)])
    if time_to_maturity-start != 0:
        coupons += (FV*math.exp(-rate*(time_to_maturity-start)))

def PA(rate, time):
    return (1-(1/(1+rate)**time))/rate



class bond:
    def __init__(this, FV, coupon, maturity, rate):
        this.PV = sum([coupon*math.exp(-rate*i) for i in range(1, len(maturity)+1)]) + FV*math.exp(-rate*maturity)


class two_securities:
    def __init__(this, p, k1, k2):
        p = convertFrac(p)
        this.p = p
        k1, k2 = convertFrac(k1), convertFrac(k2)
        E1, E2 = elemProd(p, k1),  elemProd(p, k2)
        this.K1 = k1
        this.K2 = k2
        this.E1 = E1
        this.E2 = E2
        E_k1sq, E_k2sq = elemProd(p, elemSq(k1)), elemProd(p, elemSq(k2))
        var_k1, var_k2 = E_k1sq - E1*E1,   E_k2sq - E2*E2
        sigmak1, sigmak2 = fractions.Fraction(math.sqrt(var_k1)),  fractions.Fraction(math.sqrt(var_k2))
        this.sigma1 = sigmak1
        this.sigma2 = sigmak2
        Ek1k2 = elemProd(elemProd(k1, k2, False), p)
        this.EK12 = Ek1k2
        c12 = Ek1k2 - E1*E2
        this.c12 = c12
        p12 = c12 / (sigmak1*sigmak2)
        w1 = (var_k2 - c12) / (var_k1 + var_k2 - 2*c12)
        w2 = 1-w1
        wfraction = w1/w2
        wfraction2 = w2/w1
        answers = [E1, E2, E_k1sq, E_k2sq, var_k1, var_k2, sigmak1, sigmak2, Ek1k2, c12, p12, w1, w2, wfraction, wfraction2]
        equations = ["E[K1]", "E[K2]", "E[K1^2]", "E[K2^2]", "E(K1^2)- E[K1]^2","E(K2^2)- E[K2]^2", "sqrt(Var(K1))", "sqrt(Var(K2))",
                    "E[K1*K2]", "Cov(K1, K2)", "p12", "w1", "w2", "stdDev(K1)/stdDev(K2)", "stdDev(K2)/stdDev(K1)"]
        stringVal = ["E[K1]", "E[K2]", "E[K1^2]", "E[K2^2]", "Var(K1)","Var(K2)", "StdDev(K1)", "StdDev(K2)",
                    "E[K1*K2]", "Cov(K1, K2)", "p12", "w1", "w2", "stdDev(K1)/stdDev(K2)", "stdDev(K2)/stdDev(K1)** "]
        decimalAnswers = convertBack(answers)
        dataLists = [stringVal, equations, decimalAnswers]
        df = pd.DataFrame(dataLists, index=["Variable", "Formula", "Value"]).T
        print("")
        checkForShortSelling(sigmak1, sigmak2, p12)
        display(df)
        print("**if std(K1)>std(K2)")

    def find_W_given_E(this, Ev):
        print("\nconvert w1 to (1- w2)")
        w2 = (Ev - this.E1)/(this.E2-this.E1)
        print(f"w1 is {1-w2}, w2 is {w2}")
        return (1-w2, w2)

    def portfolioRisk_given_W(this, W):
        print("Var(Kv) = (w1*sigma1)^2 + (w2*sigma2)^2 + 2* w1 * w2 (E[k1*K2] - E[K1]*E[K2]")
        Kv_var = (W[0]*this.sigma1)**2 + (W[1]*this.sigma2)**2 + 2*W[0]*W[1]*this.c12
        print(f"Var(Kv) = {Kv_var}")
        std_Kv = math.sqrt(Kv_var)
        print(f"StdDev(Kv) = {std_Kv}")
        return (Kv_var, std_Kv)


class multiple_security:
    def __init__(this, p, security_returns):
        this.p = np.array(p)
        this.Ks = np.array(security_returns)
        this.means = this.calculateMeans()
        kstr = ["K"+str(i) for i in range(1, len(this.Ks)+1)]
        stdstr = ["Std"+str(i) for i in range(1, len(this.Ks)+1)]
        print(f"mean of {kstr} is {this.means}\n")
        this.Vars, this.stds = this.calculateRisks()
        print(f"\nstd.dev: {stdstr} is {this.stds}")

    def calculateMeans(this):
        return [elemProd(this.p, security) for security in this.Ks]

    def calculateRisks(this):
        counts = len(this.Ks)

        Vars = [Variance_classic(K, this.p) for K in this.Ks]
        Stds = [math.sqrt(i) for i in Vars]
        return (Vars, Stds)

class StaData:
    def __init__(this, p, vals):
        """input: probability, vals
        """
        this.mean = elemProd(p, vals)
        this.var = Variance_classic(vals, p)
        this.stdDev = math.sqrt(this.var)



class portfolio_MVP:
    def __init__(this, m, stdDevs, correlation):
        """Given m, stdevs and correlation of the securties in the portfolio,
        this function prints: C, C^-1, M, M^-1, E[Kv], wCwT = Var(Kv), and stdev(Kv)"""
        this.m = np.array(m)
        this.u = np.array([1 for _ in m])
        this.stdDevs = stdDevs
        this.correlation = correlation
        this.a = -1
        this.calculate_rest()

    def calculate_rest(this):
        print("m is:", this.m, "\nu is:", this.u)
        this.CMatrix = np.array(makeCMatrix(this.correlation, this.stdDevs))
        this.CInverse = np.linalg.inv(this.CMatrix)
        print("\nhere is the inverse of C:\n", this.CInverse)
        print("\ncalculating M:")
        this.Mmatrix = makeM(this.m, this.CInverse, this.u)
        print("\nhere is M inverse:")
        this.MmatrixInv = np.linalg.inv(this.Mmatrix)
        print(this.MmatrixInv)
        this.minimumWeights = minWeights(this.u, this.CInverse)
        print("\nThe Expected return of the MVP is:")

        this.Emin = elemProd(this.minimumWeights, this.m)
        print(this.Emin)
        print("\nVariance is:")
        this.wCwT = find_wCwT(this.minimumWeights, this.CMatrix)
        print("wCwT = Varinace = ")
        print(this.wCwT)
        this.stdDevMin = math.sqrt(this.wCwT)
        print("\nThe Std.Dev of the portfolio is:")
        print(this.stdDevMin)

    def E_Var_std(this, weights):
        E = elemProd(weights, this.m)
        print(f"expected value is {E}")
        print("\nVariance is:")
        variance = find_wCwT(weights, this.CMatrix)
        print("wCwT = Varinace = ", variance)
        stdDev = math.sqrt(variance)
        print("\nThe Std.Dev of the portfolio is:")
        print(stdDev)
        return [E, variance, stdDev]

    def W_MVP_Given_Ev(this, Ev):
        print("*"*20)
        if this.a == -1:
            this.calculate_ab()
        print("\na, b doesnt change for each portfolio on the minimum variance line")
        print("to get the weights that satisfy a specific E[Kv], \nwe solve the linear equation: w = mu * a + b")
        this.W_given_EKv = this.compute_W_given_E(Ev)
        print("\nVariance is:")
        this.wCwT_given_EKv = find_wCwT(this.W_given_EKv, this.CMatrix)
        print("wCwT = Varinace = ")
        print(this.wCwT_given_EKv)
        this.stdDevMin_given_EKv = math.sqrt(this.wCwT_given_EKv)
        print("\nThe Std.Dev of the portfolio is:")
        print(this.stdDevMin_given_EKv)

        print("*"*20)

    def calculate_ab(this):
        print("\ncalculating for a and b:")
        print("[a, b].T = M^-1 * [m, u].T * C^-1")
        this.matrix_ab = np.matmul(np.matmul(this.MmatrixInv, np.array([this.m, this.u])), this.CInverse)
        this.a = this.matrix_ab[0]
        this.b = this.matrix_ab[1]
        print("this is vector a: ", this.a)
        print("this is vector b: ", this.b)

    def compute_W_given_E(this, Ev):
        W_given_EKv = (Ev*this.a) + this.b
        print(f"w = {W_given_EKv}")
        return W_given_EKv

    def calculate_MVL_slope(this, x, y):
        print("\nfor mu = 0:")
        w0 = this.compute_W_given_E(0)
        print("\nfor mu = 0.1:")
        w01 = this.compute_W_given_E(0.1)
        slope = this.compute_slope(w0, w01, x, y)

        # fixed index to be 0 based
        intersect = w0[y-1] - slope*w0[x-1]

        print(f"\nequation for the line on plane (x var: w{x}, y var: w{y}):")
        print(f" = {slope} * w{x} + {intersect}")

        xVal = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        yVal = np.array([(slope*xi + intersect) for xi in xVal])

        xy_table = pd.DataFrame([xVal, yVal], index=["x axis", "y axis"])
        print("sample points:")
        print(xy_table)

    def compute_slope(this, w0, w1, x, y):
        # modifying the indecies
        i, j = x-1, y-1
        print(f"\nmu=0,   w{x} = {w0[i]}")
        print(f"mu=0,   w{y} = {w0[j]}")
        print(f"mu=0.1, w'{x} = {w0[i]}")
        print(f"mu=0.1, w'{y} = {w0[j]}")
        slope = (w1[j] - w0[j]) / (w1[i] - w0[i])
        print(f"\nequation for slope: slope = (w{y} - w'{y}) / (w{x} - w'{x})")
        print(f"slope = {slope}, for graph with (x var: w{x}, y var: w{y})")
        return slope

    def market_portfolio(this, R):
        print("*"*20)
        mru = (this.m - R*this.u)
        print("\nweights in a market portfolio = ((m-R*u) * C^-1)/((m-R*u)* C^-1 * uT)")
        print(f"intermediate steps: m-R*u = {mru}")
        print("(m-R*u)* C^-1 = ", np.matmul(mru,this.CInverse))
        this.market_weights = np.matmul(mru,this.CInverse) / np.matmul(np.matmul(mru,this.CInverse), this.u.T)
        print(f"\nmarket portfolio weights = {this.market_weights}")
        this.E_Var_std(this.market_weights)


def main():

    mu = [0.08, 0.1, 0.06]
    stdevs = [0.15, 0.05, 0.12]
    corr = [
        [[1, 2],0.3],
        [[2,3], 0.0],
        [[1, 3], -0.2]
        ]
    risk_free_rate = 0.05
    x = portfolio_MVP(mu, stdevs, corr)
    weights = [-0.0182, 0.9677, 0.0505]
    x.E_Var_std(weights)


if __name__ == "__main__":
    main()