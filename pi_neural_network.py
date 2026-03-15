import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Neural Network Discovering Pi", layout="wide")

st.title("Neural Network Discovering π")

st.write(
"""
This demo trains a **neural network** to learn the boundary of a **unit circle**.

Using **Monte Carlo sampling**, we estimate π based on how many random points
fall inside the circle.
"""
)

# Sidebar controls
st.sidebar.header("Simulation Controls")

samples = st.sidebar.slider("Training samples", 1000, 50000, 10000)
hidden = st.sidebar.slider("Hidden neurons", 5, 100, 30)
test_points = st.sidebar.slider("Monte Carlo test points", 5000, 200000, 50000)

# Generate training data
x = np.random.uniform(-1,1,samples)
y = np.random.uniform(-1,1,samples)

X = np.vstack((x,y)).T

labels = (x**2 + y**2 <= 1).astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.2)

# Train neural network
model = MLPClassifier(
    hidden_layer_sizes=(hidden,hidden),
    activation='relu',
    max_iter=300
)

model.fit(X_train,y_train)

# Monte Carlo estimation
xt = np.random.uniform(-1,1,test_points)
yt = np.random.uniform(-1,1,test_points)

test_data = np.vstack((xt,yt)).T

pred = model.predict(test_data)

inside = pred==1

pi_est = 4*np.sum(inside)/test_points

# Layout
col1,col2 = st.columns(2)

# Monte Carlo visualization
with col1:

    st.subheader("Monte Carlo Simulation")

    fig1,ax1 = plt.subplots(figsize=(5,5))

    ax1.scatter(xt[inside],yt[inside],s=3,label="Inside")
    ax1.scatter(xt[~inside],yt[~inside],s=3,label="Outside")

    ax1.set_title("Random Points Classification")
    ax1.legend()

    st.pyplot(fig1)

# Decision boundary
with col2:

    st.subheader("Neural Network Decision Boundary")

    grid_size = 200

    x_vals = np.linspace(-1,1,grid_size)
    y_vals = np.linspace(-1,1,grid_size)

    xx,yy = np.meshgrid(x_vals,y_vals)

    grid = np.c_[xx.ravel(),yy.ravel()]

    pred_grid = model.predict_proba(grid)[:,1]

    pred_grid = pred_grid.reshape(grid_size,grid_size)

    fig2,ax2 = plt.subplots(figsize=(5,5))

    contour = ax2.contourf(xx,yy,pred_grid,levels=50)

    ax2.set_title("Learned Circle Boundary")

    st.pyplot(fig2)

# π convergence simulation
st.subheader("π Convergence Simulation")

steps = 2000

pi_vals = []

inside_count = 0

for i in range(1,steps+1):

    px = np.random.uniform(-1,1)
    py = np.random.uniform(-1,1)

    if px**2 + py**2 <= 1:
        inside_count += 1

    pi_vals.append(4*inside_count/i)

fig3,ax3 = plt.subplots()

ax3.plot(pi_vals,label="Estimated π")
ax3.axhline(np.pi,color='red',linestyle='--',label="True π")

ax3.set_title("Convergence to π")
ax3.legend()

st.pyplot(fig3)

# Metrics
st.subheader("π Estimation")

col3,col4,col5 = st.columns(3)

col3.metric("Estimated π", round(pi_est,6))
col4.metric("Actual π", round(np.pi,6))
col5.metric("Absolute Error", round(abs(np.pi-pi_est),6))

# Explanation
st.write(
"""
### How This Works

1️⃣ Random points are generated in a square [-1,1] × [-1,1].

2️⃣ A neural network learns whether points lie inside a circle.

3️⃣ The fraction of points inside the circle approximates the area ratio.

4️⃣ From that ratio we compute π.

This demonstrates how **machine learning + geometry + probability**
can recover a fundamental mathematical constant.
"""
)
