
from operator import itemgetter
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import pickle
from streamlit_agraph.config import Config, ConfigBuilder
from annotated_text import annotated_text
import numpy as np
import extra_streamlit_components as stx



digits      = [ i for i in range(10,14) ] 
digit_key   = lambda digit : ('attn_out', 1, None, int(digit) ) 
options = {"1000s digit":10, "100s digit":11 , "10s digit":12, "1s digit":13 }

def title(key) :
    if key[2] is None :
        return f'{key[0]} (layer {key[1]}, word {key[3]})'
    else : 
        return f'{key[0]} (layer {key[1]}, head {key[2]}, word {key[3]})'

def desc(ex, digit ) :
    if (digit<13) :
        return f"{ex[digit-9]}+{ex[digit-4]}+{ex[digit+2][1]}={ex[digit+1][0]}"
    else :
        return f"{ex[digit-9]}+{ex[digit-4]}={ex[digit+1][0]}"


group   = lambda key : f"L{key[1]}H{key[2]}"

with open( f"examples.pkl", 'rb') as file:
    examples = pickle.load(file)

if 'ex' not in st.session_state:
    st.session_state['ex']    = 0
    st.session_state['digit'] = None


st.title("Interpretability through clustering")
st.markdown("**How are activations in a transformer clustered together and what can we learn?**  \nActivations below are from a 2 layer, 3 head, 48 dimension transformer trained to solve 4 digit addition.")

st.image("4digit.png", use_column_width=True)

st.markdown("Follow the calculation to see how activations are grouped together to form token predictions.")

st.divider()


if st.button("Generate an example", type="primary") :
    st.session_state['ex']    = np.random.randint(1,len(examples))
    st.session_state['digit'] = None


example, similarities, dependency = examples[ st.session_state['ex'] ]
    
annotated_text( example[1:] )

key_to_int = {key: index for index, key in enumerate(dependency.keys())}
int_to_key = {key_to_int[key]: key for key in key_to_int.keys() }


def group(key,digit) :
    digit = int(digit)
    ex = similarities[key][1][0]
    on_path = (len(ex[digit-9])> 1) or (len(ex[digit-4]) > 1) or (len(ex[digit+1]) > 1)
    if (digit<13) :
        on_path = on_path or (len(ex[digit+2]) > 1) 

    return  1 if on_path else 0

def size(key,digit) :
    return 40 if group(key,digit) else 10

def create_nodes_edges( digit ) :

    digit_key_start =  digit_key(digit)

    nodes = { key : Node(id=key_to_int[key],  title=title(key), label=similarities[key][0], group=group(key,digit), shape="dot",size=size(key,digit)) for key in similarities.keys()  }
    nodes_used = set()

    edges = []
    def add_edge( key ) :
        for dependency_key in dependency[key] :
            edges.append( Edge( source=key_to_int[key], label=" ", target=key_to_int[dependency_key]) )
            nodes_used.add(nodes[dependency_key])
            add_edge( dependency_key )    

    nodes_used.add( nodes[digit_key_start] )
    add_edge( digit_key_start  )

    return list(nodes_used), edges

tab_data = [ stx.TabBarItemData(id=digit, title=opt, description=desc(example, digit ) ) for opt, digit in zip(options.keys(),digits) ] 
digit    = stx.tab_bar( data=tab_data, default=digits[0])



# option = st.radio('Trace path for:', options=options.keys(), horizontal=True)
# digit = options[option]

if 'digit' not in st.session_state or st.session_state['digit'] != digit or 'nodes' not in st.session_state :
    st.session_state['digit'] = digit
    nodes, edges = create_nodes_edges( digit ) 
    st.session_state['nodes'] = nodes
    st.session_state['edges'] = edges
    st.session_state['config'] = Config(from_json="config.json") 

col1, col2 = st.columns(2)

with col1 :
    st.caption('Large circles are activations with relevant features.  \n:blue[Hover] to see node details (transformer step).  \n:blue[Zoom] to see patterns in the activations.  \n:blue[Click] to see closest examples in activation space)')
    node_id = agraph(nodes=st.session_state['nodes'], edges=st.session_state['edges'], config=st.session_state['config']) 
    if node_id is None :
        node_id = key_to_int[digit_key(digit)]
    key = int_to_key[node_id]

with col2 :

    st.info( title(key) )
    # annotated_text( example[1:] )
    st.markdown(f'**Closest other examples (in order):**')
    exs = similarities[key][1]
    for ex in exs :
        annotated_text(ex[1:])


st.caption('A transformer visualized as a graph of activations. Transformer lens : pattern, result and attn_out stages are shown. Graph edges are formed where attention pattern is greater than .1, or where head specific results are summed to form attn_out.  Edges were pruned if mean ablation showed that they did not change the output.  The calculation proceeds from bottom to top (e.g.  graph depencdency is from top to bottom). The example set is formed from 5000 random additions.')

st.link_button("reference on transformer steps", "https://arena-ch1-transformers.streamlit.app/~/+/Reference_Page#diagrams", help=None, type="secondary", disabled=False, use_container_width=False)

st.link_button("details on 4 digit addition", "https://arena-ch1-transformers.streamlit.app/~/+/Monthly_Algorithmic_Problems#monthly-algorithmic-challenge-november-2023-cumulative-sum", help=None, type="secondary", disabled=False, use_container_width=False)

