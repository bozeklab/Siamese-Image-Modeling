import io
import base64
import pickle

from PIL import Image
import numpy as np

from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from sklearn.manifold import TSNE


repr_path = 'representations/data.pkl'


# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def load_representations():
    with open(repr_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the data
data = load_representations()
images = data['crops']
embeddings = data['X']
labels = data['y']


# t-SNE Outputs a 3 dimensional point for each image
tsne = TSNE(
    random_state=123,
    n_components=3,
    verbose=1,
    learning_rate=200,
    perplexity=50,
    n_iter=2000) \
    .fit_transform(embeddings)

tumor_categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

colors_per_class = ['gray', 'yellow', 'lightcoral', 'sienna', 'pink', 'purple', 'royalblue', 'springgreen',
                    'maroon', 'black', 'orange', 'indigo', 'hotpink', 'lavender', 'red']


color_map = dict(zip(tumor_categories, colors_per_class))

colors = [color_map[label] for label in labels]

fig = go.Figure(data=[go.Scatter3d(
    x=tsne[:, 0],
    y=tsne[:, 1],
    z=tsne[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=colors,
    )
)])

fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)

fig.update_layout(template='plotly_dark')

app = Dash(__name__)

app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(id="graph-5",
                  figure=fig, clear_on_unhover=True,
                  style={'width': '100vw',
                         'height': '100vh'}),
        dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
    ],
    style={'width': '100vw',
           'height': '100vh'}
)


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    mylabels = np.unique(colors)
    for l in range(len(mylabels)):
        mylabels[l] = mylabels[l].replace('_', ' ')

    # create a scatter plot.
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    for id, color in enumerate(np.unique(colors)):
        print(color)
        idx = np.array(labels) == color

        sc = ax.scatter(x[idx, 0], x[idx, 1], lw=0, s=8,
                        c=palette[id], label=mylabels[id])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
               ncol=4, prop={'size': 15}, fancybox=True, shadow=True)
    plt.xticks(rotation=90)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    #for i in range(num_classes):

        # Position of each label at median of data points.

    #    xtext, ytext = np.median(x[colors == i, :], axis=0)
    #    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #    txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #        PathEffects.Normal()])
    #   txts.append(txt)

    return f, ax, sc, txts

@app.callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
)

def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    im_matrix = images[num]
    im_url = np_image_to_base64(im_matrix)
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P(str(labels[num]), style={'font-weight': 'bold'})
        ])]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=True,  host='0.0.0.0')