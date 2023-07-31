import streamlit as st
import pandas as pd

st.header("Train Data")

st.dataframe(pd.read_csv('X_train_v1.csv'))

st.subheader("Model")
st.write(
    "Next we will create model like the image below. The image are taken from [here](https://arxiv.org/abs/1510.03820): ")

st.image('cnn.png')

st.write('''
    Illustration of a Convolutional Neural Network (CNN) architecture for sentence classification. 
    Here we depict three filter region sizes: 2, 3 and 4, each of which has 2 filters.
    Every filter performs convolution on the sentence matrix and generates (variable-length) feature maps. 
    Then 1-max pooling is performed over each map, i.e., the largest number from each feature map is recorded. 
    Thus a univariate feature vector is generated from all six maps, and these 6 features are concatenated to form a feature vector for the penultimate layer. 
    The final softmax layer then receives this feature vector as input and uses it to classify the sentence; 
    here we assume binary classification and hence depict two possible output states. 

''')

st.write("Here is the model implementation in Python:")
st.code('''
inputs = Input(shape=(input_length,))

embedding = Embedding(input_dim=input_dim,
                        output_dim=output_dim, input_length=input_length)(inputs)

conv1 = Conv1D(filters=2, kernel_size=2, activation='relu')(embedding)
conv2 = Conv1D(filters=2, kernel_size=3, activation='relu')(embedding)
conv3 = Conv1D(filters=2, kernel_size=4, activation='relu')(embedding)

concat = Concatenate()([GlobalMaxPooling1D()(
    conv1), GlobalMaxPooling1D()(conv2), GlobalMaxPooling1D()(conv3)])

dense = Dense(units=6, activation='relu')(concat)

outputs = Dense(units=2, activation='softmax')(dense)

model = Model(inputs=inputs, outputs=outputs)
''')
