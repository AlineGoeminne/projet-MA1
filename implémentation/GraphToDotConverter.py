from GraphGame import Graph
from GraphGame import Vertex
from GraphGame import ReachabilityGame

def minMax_graph_to_dot(game,value, file_name):
    graph = game.graph
    target = game.goal
    with open(file_name,"w") as file:
        file.write("digraph G { \n")
        for v in graph.vertex:
            if v.player == 1:
                output = "v"+str(v.id)+"[label = \"v"+str(v.id)+" | "+str(value[v.id])+ "\", shape=circle];\n"
                file.write(output)
            else:
                output = "v" + str(v.id) + "[label = \"v"+str(v.id)+" | "+str(value[v.id])+ "\", shape=square];\n"
                file.write(output)
            if v.id in target:
                output = "v"+str(v.id)+" [color = yellow, style = filled] \n"
                file.write(output)
        for v_list_index in xrange(len(graph.succ)):
            for s_tuple in graph.succ[v_list_index]:
                output ="v"+str(v_list_index)+" ->  v"+str(s_tuple[0])+" [label = "+str(s_tuple[1])+" ]\n"
                file.write(output)


        file.write("}\n")
        file.close()
def graph_house_to_dot(game,file_name):
    graph = game.graph
    target = game.goal
    with open(file_name,"w") as file:
        file.write("digraph G{ \n")
        for v in graph.vertex:
            output = "v"+str(v.id)+"[label = \" " + str(v) + "\",shape = rectangle];\n"
            file.write(output)

        for t in target:
            for g in t:
                output = "v" + str(g) + " [color = yellow, style = filled] \n"
                file.write(output)


        for v_list_index in xrange(len(graph.succ)):
            for s_tuple in graph.succ[v_list_index]:
                output = "v" + str(v_list_index) + " ->  v" + str(s_tuple[0]) + " [label = \" " + str(s_tuple[1]) + "\" ]\n"
                file.write(output)
        file.write("}\n")
        file.close()



if __name__ == '__main__':

    W = 10

    v1 = Vertex(1,2)
    v2 = Vertex(2,1)
    v3 = Vertex(3,2)

    vertex = [v1, v2, v3]

    succ1 = [(2,-1),(3,-W)]
    succ2 = [(1,0),(3,0)]
    succ3 = [(3,0)]

    succ = [succ1,succ2,succ3]

    target = {3}

    graph = Graph(vertex, None , None, succ, W)
    game = ReachabilityGame(2, graph, None, target, None)



    minMax_graph_to_dot(game,"test.dot")
