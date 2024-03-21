import streamlit
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_agraph.config import Config, ConfigBuilder

nodes = []
edges = []
nodes.append( Node(id="Spiderman", 
                   label="Peter Parker", 
                   size=25, 
                   shape="dot",
                   group=str(1)) )
nodes.append( Node(id="Captain_Marvel", 
                   size=25,
                   shape="dot",
                   group=str(2)))
edges.append( Edge(source="Captain_Marvel", 
                   label="friend_of", 
                   target="Spiderman", 
                   # **kwargs
                   ) 
            ) 




# 1. Build the config (with sidebar to play with options) .
config_builder = ConfigBuilder(nodes)
config = config_builder.build()



return_value = agraph(nodes=nodes, 
                      edges=edges, 
                      config=config)

config.save("config.json")