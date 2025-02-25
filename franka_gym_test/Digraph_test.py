import matplotlib.pyplot as plt
import networkx as nx

class GraphDrawer:
    def __init__(self, graph):
        self._graph = graph

    def draw(self):
        # Verify graph is not empty
        if not self._graph.nodes or not self._graph.edges:
            print("Graph is empty.")
            return
        
        options = {
            'node_color': 'blue',
            'node_size': 100,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        
        # Layout and drawing
        pos = nx.spring_layout(self._graph)
        nx.draw(self._graph, pos, with_labels=True, **options)

        # Ensure rendering
        plt.draw()
        plt.show()

# Example usage
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 4)])
drawer = GraphDrawer(G)
drawer.draw()
