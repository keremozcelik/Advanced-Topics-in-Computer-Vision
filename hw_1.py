# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:51:59 2020

@author: Ben
"""
#%%import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

#%% Read image and RGB to Grayscale
img = Image.open("SunnyLake.bmp") #bmp formatında
RGB_img = np.asarray(Image.open("SunnyLake.bmp")) #int array formatında
Gray_func = np.asarray(img.convert('L')) #hazır fonksiyon ile grayscale çevirme
SP_img = np.asarray(Image.open("SP_Noisy_SunnyLake.png"))

R = RGB_img[:,:,0]
G = RGB_img[:,:,1]
B = RGB_img[:,:,2]
I = (np.rint(0.2989 * R + 0.5870 * G + 0.1140 * B)).astype(int) #uygun katsayılarla grayscale.np.rint ile en yakın int'e çevrilip int olarak tutuldu.
I2 = (R + G + B) / 3 #RGB kanallarının ortalmasını alarak grayscale

#%%Ploting
plt.figure(1)
plt.subplot(1,2,1) , plt.imshow(RGB_img) ,  plt.title("RGB Image")
plt.subplot(1,2,2) , plt.imshow(I,cmap="gray"), plt.title("Gray Scaled Image")

plt.figure(2)
plt.subplot(1,3,1) , plt.imshow(R), plt.title("Red Channel")
plt.subplot(1,3,2) , plt.imshow(G), plt.title("Green Channel")
plt.subplot(1,3,3) , plt.imshow(B), plt.title("Blue Channel")

#%% Histogram
histogram_array = np.zeros(256, dtype=int)
count = 0;
N = np.arange(256)

for i in range(0,256):
    for j in range(I.shape[0]):
        for k in range(I.shape[1]):
            if I[j][k] == i:
                count = count+1;
    histogram_array[i] = count
    count = 0

plt.figure(3)
plt.subplot(1,2,1), plt.plot(N,histogram_array), plt.axis([0, 256, 0, 4000])
plt.xlabel("Intensity Levels") , plt.ylabel("Number of Pixels") ,plt.title("Histogram of the Image(without python function)")

plt.subplot(1,2,2), plt.hist(I.ravel(),256,[0,256]), plt.show()
plt.xlabel("Intensity Levels"), plt.ylabel("Number of Pixels"), plt.title("Histogram of the Image(with python function)")

#%% Automatic Thresholding
T = []

# Step 1 = Compute Mean = init threshold
total = np.cumsum(histogram_array)
intensity =np.sum(N*histogram_array)
T.append(int(intensity/total[-1])) #İlk kaba threshold mean ile
            
# Step 2 = Compute Below and Above of init threshold and new threshold       
below_total = np.cumsum(histogram_array[0:int(T[0])+1])     #T[0] değerinin altındaki bölge için alt threshold hesaplanır.
below_intensity = np.sum(N[0:int(T[0])+1]*histogram_array[0:int(T[0])+1])

above_total = np.cumsum(histogram_array[int(T[0]):])     #T[0] değerinin üst bölgesi için üst threshold hesaplanır.
above_intensity = np.sum(N[int(T[0]):]*histogram_array[int(T[0]):])

MBT = below_intensity / below_total[-1]   #mean below threshold
MAT = above_intensity / above_total[-1]   #mean above threshold 
T.append(int((MBT+MAT)/2))   #yeni threshold = T[1]        

# Step 3 = repeat step 2 while T[i]- T[i-1] < 1   Threshold  bir sabit değere oturuncaya kadar.
index = 1
while abs(T[index]-T[index-1])>=1 : 
    tot= np.cumsum(histogram_array[0:int(T[index])+1])
    MBT2=np.sum(N[0:int(T[index])+1]*histogram_array[0:int(T[index])+1])/tot[-1];
    
    tot2=np.cumsum(histogram_array[int(T[index]):])
    MAT2=np.sum(N[int(T[index]):]*histogram_array[int(T[index]):])/tot2[-1];
    
    index=index+1;
    T.append(int((MAT2+MBT2)/2)); 
    Threshold=T[-1];  # T listesinin son elemanı Threshold değeri olur.
    
#%% foreground and background separation with threshold value
for j in range(I.shape[0]):
    for k in range (I.shape[1]):
        if I[j][k] >= Threshold:
                I[j][k]=0 #Background = black
        else:
                I[j][k]=255 #Foreground = white

plt.figure(4), plt.imshow(I,cmap="gray"), plt.title('Foreground & Background with Automatic Thresholding')

#%% Gaussian Noise

def add_gauss_noise(img,mean,sigma_list=[],value=None):
    noisy_img = []

    for sigma in sigma_list:
        gauss = np.random.normal(mean, sigma, img.shape) #gauss gürültü üretme 
        noisy_img.append((img+gauss).astype(int))  #farklı sigmalara ait gürültülü resimler listeye eklendi.
    
    plt.figure()
    for i in range(1,len(sigma_list)+1):
        if value == None:
            plt.subplot(2,np.ceil(len(sigma_list)/2),i), plt.imshow(noisy_img[i-1]), plt.title('Gaussian Noised Sigma = {}'.format(sigma_list[i-1]))
        else:
            plt.subplot(2,np.ceil(len(sigma_list)/2),i), plt.imshow(noisy_img[i-1],cmap="gray"), plt.title('Gaussian Noised Sigma = {}'.format(sigma_list[i-1]))
    return noisy_img
    
add_gauss_noise(RGB_img,0,sigma_list=[5,10,20,30]) #RGB resim için gauss noise
add_gauss_noise(R,0,sigma_list=[5,10,20,30]) # Red kanal için gauss noise
add_gauss_noise(G,0,sigma_list=[5,10,20,30]) # Green kanal için gauss noise
add_gauss_noise(B,0,sigma_list=[5,10,20,30]) # Blue kanal için gauss noise
add_gauss_noise(I,0,sigma_list=[5,10,20,30],value=1) #Black-White resim için gauss noise

#%% Mean Filter = Low Pass Filter

def mean_filter(img,name,kernel_size=[],value=None): #value değeri verilirse gray colormapi ile çizilir. verilmezse renkli colormap kullanılır.
    mean_img =[]
    
    for k in kernel_size:
        mean_img.append(cv2.blur(img,(k,k)))
    
    plt.figure() 
    if value == None:
        plt.subplot(2,np.ceil((len(kernel_size)/2)+0.5),1), plt.imshow(img),plt.title("{}".format(name)) #ilk kısma fonksiyona verilen image çizilir
        for i in range(1,len(kernel_size)+1):  #subplotun geri kalan kısımlarına kernel_size a göre mean filter sonuçları çizilir.
            plt.subplot(2,np.ceil((len(kernel_size)/2)+0.5),i+1), plt.imshow(mean_img[i-1]),plt.title("Mean Filter kernel size:{}*{}".format(kernel_size[i-1],kernel_size[i-1]))
    else:
        plt.subplot(2,np.ceil((len(kernel_size)/2)+0.5),1), plt.imshow(img,cmap="gray"),plt.title("{}".format(name)) #ilk kısma fonksiyona verilen image çizilir
        for i in range(1,len(kernel_size)+1): #subplotun geri kalan kısımlarına kernel_size a göre mean filter sonuçları çizilir.
            plt.subplot(2,np.ceil((len(kernel_size)/2)+0.5),i+1), plt.imshow(mean_img[i-1],cmap="gray"),plt.title("Mean Filter kernel size:{}*{}".format(kernel_size[i-1],kernel_size[i-1]))
    return mean_img
      
mean_filter(RGB_img,"RGB Image",kernel_size=[3,5,9]) #Normal RGB image için
mean_filter(I,"BI Image",kernel_size=[3,5,9],value=1) #BI image için

#Gauss noise sigma=30 içeren resimlere mean filter uygulaması
mean_filter(add_gauss_noise(RGB_img,0,sigma_list=[5,10,20,30])[3],"Gaussian Noised Sigma = 30 RGB Image",kernel_size=[3,5,9]) #Sigma =30 gaussian noise içeren RGB image için
mean_filter(add_gauss_noise(I,0,sigma_list=[5,10,20,30],value=1)[3],"Gaussian Noised Sigma = 30 BI Image",kernel_size=[3,5,9],value=1)

#%% Laplacian Filter = High Pass Filter
sigma_list=[0,0.5,1,3] #düşük sigmalarda sonuç daha iyi görülür.
I_L = np.uint8(add_gauss_noise(I,0,sigma_list,value=1)) #gauss gürültü eklenmiş BI resimleri listesi döner.

plt.figure()
for i in range(1,len(sigma_list)+1):
    laplacian_img = cv2.Laplacian(I_L[i-1], cv2.CV_8U, (5,5)) #gauss gürültülü BI resimlerinin sırayla laplacian filterdan geçirilmesi.
    plt.subplot(2,np.ceil(len(sigma_list)/2),i),plt.imshow(laplacian_img,cmap="gray"), plt.title("Laplacian Filter using on BI_sigma={} Image".format(sigma_list[i-1]))

#%% Median Filter on Salt-Pepper Noised Image
median_kernel = [3,5,9] #kernel listesinin boyutu değiştirilebilir.
clear_img = []

for i in median_kernel:
    clear_img.append(cv2.medianBlur(SP_img,i))

plt.figure()
plt.subplot(2,np.ceil((len(median_kernel)/2)+0.5),1), plt.imshow(SP_img),plt.title("Salt-Paper Noised Image") #median_kernel uzunluğuna göre subplot ayarlanır.
for i in range(1,len(median_kernel)+1):
    plt.subplot(2,np.ceil((len(median_kernel)/2)+0.5),i+1), plt.imshow(clear_img[i-1]),plt.title("Clear Image Median Filter kernel:{}*{}".format(median_kernel[i-1],median_kernel[i-1]))