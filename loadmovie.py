
import skimage.io
import skimage.viewer as skview

filename='movie.tif'
data = skimage.io.imread(filename)
for k in range(64):
    frame= data[k,:,:,:]
    print('dimension frame',frame.shape)
    # tranpose 5,2,512,512 5 to 5,512,512,2
    transpose_frame = np.transpose(frame, (0, 2,3,1))
    print('dimension tranpose',transpose_frame.shape)

    for i in range(5):
        datastack = transpose_frame[i, :, :, :]
        print(datastack.shape) #(512, 512, 2)
        #skimage.io.imsave('movie_stack%d.png' % i, datastack)
        for t in range(2):
            channel_split = datastack[:, :, t]
            zeros = np.zeros((512, 512), dtype=np.int8)
            if t == 0:  # red channel
                channel_split = np.stack((channel_split, zeros, zeros), axis=2)
                print(channel_split.shape)
                skimage.io.imsave("movie_frame%d_stack%d,channel%d.png" % (k,i, t), channel_split)
            else:  # blue channel
                channel_split = np.stack((zeros, zeros,channel_split), axis=2)
                print(channel_split.shape)
                skimage.io.imsave("movie_frame%d_stack%d,channel%d.png" % (k,i, t), channel_split)
