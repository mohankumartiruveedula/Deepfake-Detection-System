//here we are considering adjacent list representation

#include <bits/stdc++.h>
using namespace std;

//this func is related to dfsRec for getting decreasing finishing order func
void dfsRec(vector<vector<int>> &graph,stack<int> &st,int s,vector<bool> &visited){
    visited[s]=true;
    for(auto i : graph[s]){
        if(!visited[i]){
            dfsRec(graph,st,i,visited);
        }
    }
    st.push(s);
}

//this fun is used to get decreasing order finishing time of vertices
vector<int> getMyOrder(vector<vector<int>> &graph){
    vector<int> order;
    stack<int> st;
    vector<bool> visited(graph.size(),false);
    for(int i=0 ; i < graph.size() ; i++){
        if(!visited[i]){
            dfsRec(graph,st,i,visited);
        }
    }
    while(!st.empty()){
        order.push_back(st.top());
        st.pop();
    }
    return order;
}

//this func is to reverse the edges of graph and give a new graph
vector<vector<int>> reverseGraph(vector<vector<int>> &graph){
    vector<vector<int>> revGraph(graph.size());
    for(int i=0 ; i < graph.size() ; i++){
        for(auto j : graph[i]){
            revGraph[j].push_back(i);
        }
    }
    return revGraph;
}

//this is a dfs func for traversing used in kosaraju's func 
void dfs(vector<vector<int>> &graph,int s,vector<bool> &visited,vector<int> &temp){
    visited[s]=true;
    temp.push_back(s);
    for(auto i : graph[s]){
        if(!visited[i]){
            dfs(graph,i,visited,temp);
        }
    }
}

//this man of the hour kosaraju func which will give the set of sets of strongly connnected components
vector<vector<int>> kosaraju(vector<vector<int>> &graph){
    vector<vector<int>> result;

    vector<int> order = getMyOrder(graph);

    vector<vector<int>> revGraph = reverseGraph(graph);

    vector<bool> visited(graph.size(),false);

    for(auto i : order){
        vector<int> temp;
        if(!visited[i]){
            dfs(revGraph,i,visited,temp);
        }
        if(temp.size()!=0){
            result.push_back(temp);
        }
    }

    return result;
}

//driver bhai sab
int main(){
    vector<vector<int>> graph;
    int n;
    cout<<"\nEnter number of vertices : ";
    cin>>n;
    graph.resize(n);
    int choice = 0 ;
    while(choice != 2){
        cout<<"\nWould u like to enter an edge in graph ? \n1.Yes\n2.No\n";
        cin>>choice;
        switch(choice){
            case 1:
                int a,b;
                cout<<"Enter u v :";
                cin>>a>>b;
                graph[a].push_back(b);
                break;
            case 2:
                break;
            default:
                cout<<"\nSelect a valid option";
        }
    }
    vector<vector<int>> strong_set = kosaraju(graph);

    cout<<"\nStrongly Connected Component sets are \n";

    for(int i=0 ; i < strong_set.size() ; i++){
        cout<<"\nSet "<<i+1<<" is :-\n";
        for(auto j : strong_set[i]){
            cout<<j<<" ";
        }
    }
    return 0;
}