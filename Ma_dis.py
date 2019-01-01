import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = " 31/12/2018"


def generate_data(mean, cov, num=15):
    means = [mean, mean]
    covs = [[cov, 0], [0, cov]]
    x, y = np.random.multivariate_normal(means, covs, num).T
    return x, y


def gene_data(mean, num, filaname):
    x1, y1 = generate_data(mean, num, 15)
    get_mean_cov(x1, y1)
    view_data(x1, y1, filaname)


def view_data(x, y, filaname):
    x = np.array(x)
    y = np.array(y)
    data = pd.DataFrame(np.vstack((x, y)).T)
    data.to_csv(filaname)
    assert x.shape == y.shape
    form = '%.12f   %.12f'

    for i in range(x.shape[0]):
        print(form % (x[i], y[i]))

    print('x mean: {0} y mean: {1}'.format(np.mean(x), np.mean(y)))
    print('x cov: {0} y cov: {1}'.format(np.cov(x), np.cov(y)))


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


def read_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data).astype(np.float32)[:, 1:]
    return data[:, 0], data[:, 1]


def data_plot(x1, y1, x2, y2):
    plt.scatter(x1, y1, c='r')
    plt.scatter(x2, y2, c='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend(['data1', 'data2'])
    plt.show()


def test(x1, y1, filename='val1.csv'):
    dia = np.array([[1, 0], [0, 1]])
    data = []
    for i in range(x1.shape[0]):
        tv = [[x1[i]], [y1[i]]]
        mdis1 = get_mahalanobis_dis(tv, mean=[0, 0], cov=dia * 5)
        mdis2 = get_mahalanobis_dis(tv, mean=[10, 10], cov=dia * 20)
        diff = mdis1 - mdis2
        haha = 'G1' if diff <= 0 else 'G2'
        data.append([mdis1, mdis2, diff])
        print('%.8f %.8f %.8f %s' % (mdis1, mdis2, diff, haha))
    data = np.array(data).reshape(15, 3)
    data = pd.DataFrame(data)
    data.to_csv(filename)


def main():
    filename1 = 'tmp_data1.csv'
    filename2 = 'tmp_data2.csv'
    gene_data(1, 3, filename1)
    gene_data(7, 50, filename2)

    x1, y1 = read_data(filename1)
    mean1, cov1 = get_mean_cov(x1, y1)
    x2, y2 = read_data(filename2)
    mean2, cov2 = get_mean_cov(x2, y2)

    print(mean1, mean2)
    print(cov1, cov2)
    
    data_plot(x1, y1, x2, y2)

    test(x1, y1, 'val1.vsv')
    test(x2, y2, 'val2.vsv')


if __name__ == "__main__":
    main()
