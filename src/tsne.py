import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE

def tSNE(model, xTest, yTest, modelName, classes):
    intermediate_output = model.predict(xTest)

    clusters = np.array(yTest)
    data = pd.DataFrame(np.array(intermediate_output))
    data['Cluster'] = clusters

    plotX = data

    perplexity = 50
    tsne_2d = TSNE(n_components=2, perplexity=perplexity)
    TCs_2d = pd.DataFrame(tsne_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))
    TCs_2d.columns = ["TC1_2d", "TC2_2d"]
    plotX = pd.concat([plotX, TCs_2d], axis=1, join='inner')

    clusters = [plotX[plotX["Cluster"] == i] for i in range(len(classes))]
    trace_list = []

    for i in range(len(classes)):
        trace = go.Scatter(
            x=clusters[i]["TC1_2d"],
            y=clusters[i]["TC2_2d"],
            mode="markers",
            name=str(classes[i]),
            marker=dict(color=f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})'),
            text=None
        )
        trace_list.append(trace)

    layout = dict(
        xaxis=dict(title='TC1', ticklen=5, zeroline=False),
        yaxis=dict(title='TC2', ticklen=5, zeroline=False)
    )

    fig = go.Figure(dict(data=trace_list, layout=layout))
    return fig