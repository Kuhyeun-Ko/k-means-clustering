import numpy as np
import cv2
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_data(option, center1, radius1, rotation1, num_samples1, center2, radius2, rotation2, num_samples2):

    ## Check option argument
    if option is 'BS':
        assert num_samples1==num_samples2, 'No balanced'
        assert radius1[0]==radius1[1], 'No spherical'
        assert radius2[0]==radius2[1], 'No spherical'
    elif option is 'IS':
        assert num_samples1!=num_samples2, 'No imbalanced'
        assert radius1[0]==radius1[1], 'No spherical'
        assert radius2[0]==radius2[1], 'No spherical'
    elif option is 'BN':
        assert num_samples1==num_samples2, 'No balanced'
        assert radius1[0]!=radius1[1], 'No non-spherical'
        assert radius2[0]!=radius2[1], 'No non-spherical'
    elif option is 'IN':
        assert num_samples1!=num_samples2, 'No imbalanced'
        assert radius1[0]!=radius1[1], 'No non-spherical'
        assert radius2[0]!=radius2[1], 'No non-spherical'
    else: raise TypeError('No such a option.')    

    ## Initilize radius(0~radius), angle(0~2pi) and express as cartesian coordinate system
    rx1=np.random.uniform(0,radius1[0], size=num_samples1)
    ry1=np.random.uniform(0,radius1[1], size=num_samples1)
    rx2=np.random.uniform(0,radius2[0], size=num_samples2)
    ry2=np.random.uniform(0,radius2[1], size=num_samples2)

    angle1=np.pi*np.random.uniform(0,2, size=num_samples1)
    angle2=np.pi*np.random.uniform(0,2, size=num_samples2)
    
    x1=rx1*np.cos(angle1)
    y1=ry1*np.sin(angle1)
    x2=rx2*np.cos(angle2)
    y2=ry2*np.sin(angle2)

    ## Rotation and offset
    new_x1=x1*np.cos(rotation1)+y1*np.sin(rotation1)+center1[0]
    new_y1=-x1*np.sin(rotation1)+y1*np.cos(rotation1)+center1[1]
    
    new_x2=x2*np.cos(rotation2)+y2*np.sin(rotation2)+center2[0]
    new_y2=-x2*np.sin(rotation2)+y2*np.cos(rotation2)+center2[1]
    
    ## Plotting
    plt.clf()
    for i in range(num_samples1): plt.scatter(new_x1, new_y1, color='r', s=20)
    for i in range(num_samples2): plt.scatter(new_x2, new_y2, color='g', s=20)
    plt.savefig('./%s_data.png'%option)
    
    return np.vstack((np.stack((new_x1,new_y1), axis=-1), np.stack((new_x2,new_y2), axis=-1)))
   
## Hard k-means clsutering
# K: # of cluster
# D: feature dimension
def hard_kmeans(data, iteraions, K, D):

    ## Reshape and initilization
    # reshape data to input k-means clustering
    samples=data.reshape(-1,D)
    num_samples=samples.shape[0]

    # initilize mean of cluster(mu)
    mu=np.random.uniform(low=20, high=50 ,size=(K, D))
    
    ## Start EM(k-means clustering)
    for step in range(iteraions):

        ## E-step: assigning responsibilities
        r=np.zeros((num_samples, K))
        for i in range(num_samples):
            # assign to the responsibility having shortest distance
            min_idx=np.argmin(np.linalg.norm(mu-samples[i], 2, axis=1), axis=0)
            r[i, min_idx]=1

        ## Plotting only for 2D, 3D data
        plt.clf()
        # 2D data
        if D==2:
            for mu1, mu2 in mu: plt.scatter(mu1, mu2, color='r', s=50, marker='x')
            plt.scatter(samples[:,0], samples[:,1], c=r[:,1], s=20,  cmap=plt.cm.autumn)
            plt.savefig('./hard_kmeans_%s.png'%(step))
        # 3D data( ex) image )
        elif D==3:
            color_samples=np.matmul(r,mu)
            color_samples=color_samples.reshape(data.shape[0], data.shape[1],3)
            cv2.imwrite('./hard_kmeans_%s.png'%(step), color_samples)
        else: pass

        ## M-step: update mean of cluter(mu)
        for i in range(K): mu[i]=r[:,i].dot(samples) / r[:,i].sum()

## Soft k-means clsutering
# K: # of cluster
# D: feature dimension
def soft_kmeans(data, iteraions, K, D):
    
    ## Reshape and initilization
    # reshape data to input k-means clustering
    samples=data.reshape(-1,D)
    num_samples=samples.shape[0]

    # initilize mean of cluster(mu) and softness(beta)
    mu=np.random.uniform(low=20, high=50 ,size=(K, D))
    beta=1   

    ## Start EM(k-means clustering)
    for step in range(iteraions):

        ## E-step: assigning responsibilities
        r=np.zeros((num_samples, K))
        # assign to the responsibility along distance
        for i in range(num_samples): r[i]=np.exp(-beta*np.linalg.norm(mu-samples[i], 2, axis=1))
        r /=r.sum(axis=1, keepdims=True)

        ## Plotting only for 2D, 3D data
        plt.clf()
        # 2D data
        if D==2:
            for mu1, mu2 in mu: plt.scatter(mu1, mu2, color='r', s=50, marker='x')
            plt.scatter(samples[:,0], samples[:,1], c=r[:,1], s=20,  cmap=plt.cm.autumn)
            plt.savefig('./soft_kmeans_%s.png'%(step))
        # 3D data( ex) image )
        elif D==3: 
            color_samples=np.matmul(r,mu)
            color_samples=color_samples.reshape(data.shape[0], data.shape[1],3)
            cv2.imwrite('./soft_kmeans_%s.png'%(step), color_samples)
        else: pass

        ## M-step: update mean of cluter(mu)
        for i in range(K): mu[i]=r[:,i].dot(samples) / r[:,i].sum()


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--data-format', choices=['points', 'images'], required=True) #
    parser.add_argument('-o', '--options', choices=['soft', 'hard'], required=True) #
    parser.add_argument('-im', '--input-image', default='Lenna.jpg') #
    parser.add_argument('-k', '--num_clusters', default=5, type=int)
    parser.add_argument('-it', '--num_iterations', default=5, type=int)

    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = parse_args()

    ## Artificial datasets
    if args.data_format == "points":
        # make datsets
        # BS(Balanced Spherical), IS(Imbalanced Spherical), BN(Balanced Non-spherical), IN(Imbalanced Non-spherical) 
        BS_samples=make_data('BS', center1=(30,40), radius1=(10,10), rotation1=0, num_samples1=50, center2=(45,40), radius2=(10,10), rotation2=0, num_samples2=50)
        IS_samples=make_data('IS', center1=(30,40), radius1=(10,10), rotation1=0, num_samples1=50, center2=(45,40), radius2=(10,10), rotation2=0, num_samples2=30)
        BN_samples=make_data('BN', center1=(30,40), radius1=(2,10), rotation1=np.pi/3, num_samples1=50, center2=(45,40), radius2=(10,5), rotation2=0, num_samples2=50)
        IN_samples=make_data('IN', center1=(30,40), radius1=(2,10), rotation1=np.pi/3, num_samples1=50, center2=(45,40), radius2=(10,5), rotation2=0, num_samples2=30)
        
        # k-means clustering
        if args.options=="soft":
            soft_kmeans(BS_samples, args.num_iterations, K=args.num_clusters, D=BS_samples.shape[1])
        else:
            hard_kmeans(BS_samples, args.num_iterations, K=args.num_clusters, D=BS_samples.shape[1])
        
    ## Real dastsets
    else:
        
        img=cv2.imread(args.input_image, cv2.IMREAD_COLOR)
        if args.options=="soft":
            soft_kmeans(img, args.num_iterations, K=args.num_clusters, D=img.shape[2])
        else:
            hard_kmeans(img, args.num_iterations, K=args.num_clusters, D=img.shape[2])

