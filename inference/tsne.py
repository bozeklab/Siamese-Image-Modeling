import io
import base64
import pickle
import numpy as np
from PIL import Image
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go


# Helper function to convert a numpy image to base64
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


# Load the data from 'representations/data.pkl'
with open('representations/data.pkl', 'rb') as f:
    data = pickle.load(f)

images, embeddings, labels = data['crops'], data['X'], data['y']

# Create the t-SNE embedding
from sklearn.manifold import TSNE
tsne = TSNE(random_state=123, n_components=3, verbose=1, learning_rate=200, perplexity=50, n_iter=2000).fit_transform(embeddings)

dlbcl_cells = {
    0: 'plasma_cell', 1: 'eosinophil', 2: 'macrophage', 3: 'vessel',
    4: 'apoptotic_bodies', 5: 'epithelial_cell', 6: 'normal_small_lymphocyte',
    7: 'large_leucocyte', 8: 'stroma', 9: 'immune_cells', 10: 'unknown',
    11: 'erythrocyte', 12: 'mitose', 13: 'positive', 14: 'tumor'
}

colors_per_class = ['red', 'green', 'blue', 'purple', 'pink', 'cyan',
                   'orange', 'magenta', 'lightgreen', 'yellow', 'lightblue',
                   'sienna', 'lightseagreen', 'darkred', 'darkorange']

colors = [colors_per_class[int(l)] for l in labels]
labels = [dlbcl_cells[int(label)] for label in labels]

# Set up the Dash app
app = Dash(__name__)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=tsne[:, 0], y=tsne[:, 1], z=tsne[:, 2],
    mode='markers', marker=dict(size=2, color=colors)
)])

fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)

app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True,
                  style={'width': '100vw', 'height': '100vh'}),
        dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
    ],
    style={'width': '100vw', 'height': '100vh'}
)


# Callback to display image and label on hover
@app.callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData")
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    hoverPoint = hoverData["points"][0]
    num = hoverPoint["pointNumber"]

    im_matrix = images[num]
    im_url = np_image_to_base64(im_matrix)
    children = [
        html.Div([
            html.Img(src=im_url, style={"width": "50px", 'display': 'block', 'margin': '0 auto'},),
            html.P(str(labels[num]), style={'font-weight': 'bold'})
        ])]

    return True, hoverPoint["bbox"], children


if __name__ == "__main__":
    app.run_server(debug=True,  host='0.0.0.0')


