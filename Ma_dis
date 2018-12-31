import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = " 31/12/2018"


def generate_data(num, mean, cov, p=None):
    means = [mean, mean]
    covs = [[cov, 0], [0, cov]]  # diagonal covariance, points lie on x or y-axis
    x, y = np.random.multivariate_normal(means, covs, num).T
    print('x mean: {0} y mean: {1}'.format(np.mean(x), np.mean(y)))
    print('x cov: {0} y cov: {1}'.format(np.cov(x), np.cov(y)))
    if p:
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x, y, 'x')
        plt.axis('equal')
        plt.savefig('data_%d_%d.pdf' % (mean, cov))
        # plt.show()
    return x, y


def get_mahalanobis_dis(point, mean, cov):
    point = np.array(point)
    mean = np.array(mean).reshape(-1, 1)
    cov = np.array(cov)

    pinv = np.linalg.pinv(cov)
    t = np.dot((point - mean).T, pinv)
    return np.sqrt(np.dot(t, (point - mean)))


def get_mean_cov(x, y):
    data = np.vstack((np.array(x), np.array(y)))
    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    return mean, cov


def main():
    dia = np.array([[1, 0], [0, 1]])

    x1, y1 = generate_data(15, 0, 5)
    mean1, cov1 = get_mean_cov(x1, y1)

    x2, y2 = generate_data(15, 10, 20)
    mean2, cov2 = get_mean_cov(x2, y2)
    # plt.scatter(x1, y1, c='r')
    # plt.scatter(x2, y2, c='b')
    # plt.show()
    # exit()

    x = [[-2.4066752], [1.18054059]]
    mdis = get_mahalanobis_dis(x, mean1, cov1)
    odis = get_mahalanobis_dis(x, mean1, dia)
    print(mdis, odis)

    mdis = get_mahalanobis_dis(x, mean2, cov2)
    odis = get_mahalanobis_dis(x, mean2, dia)
    print(mdis, odis)

if __name__ == "__main__":
    main()
