wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip dogImages.zip  && rm dogImages.zip
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
unzip lfw.zip && rm lfw.zip && rm -rf __MACOSX
wget -P ./bottleneck_features/ https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz
