#include <bits/stdc++.h>
using namespace std;

class Graph
{
private:
    int v, e;
    vector<vector<int>> adj;
    vector<int> visited;
    void addEdge(int u, int v);

public:
    Graph(int v, int e);
    ~Graph();
    void cleanVisArr();
    void MakeGraph(bool useInBuilt);
    void DFS(int start);
    void BFS(int start);
};

// Member functions of Graph
Graph::Graph(int v, int e)
{
    this->v = v;
    this->e = e;
    adj.resize(v);
    visited.resize(v);
    cleanVisArr();
}

void Graph::addEdge(int u, int v)
{
    adj[u].push_back(v);
    adj[v].push_back(u);
}

void Graph::cleanVisArr()
{
    #pragma omp parallel for
    for (int i = 0; i < v; i++)
        visited[i] = 0;
}

void Graph::MakeGraph(bool useInBuilt)
{
    if (!useInBuilt)
    {
        int src, dest;
        for (int i = 0; i < e; i++)
        {
            cout << "Enter edge no. " << i + 1 << " : ";
            cin >> src >> dest;
            addEdge(src, dest);
        }
    }
    else
    {
        v = 4, e = 4;

        adj.resize(v);
        visited.resize(v);
        cleanVisArr();

        addEdge(0, 2);
        addEdge(0, 1);
        addEdge(2, 3);
        addEdge(1, 2);
    }
}

void Graph::DFS(int start)
{
    visited[start] = 1;
    cout << start << " -> ";

    #pragma omp parallel for
    for (auto &&i : adj[start])
    {
        if (!visited[i])
        #pragma omp task
        {
            DFS(i);
        }
    }
}

void Graph::BFS(int start)
{
    queue<int> q;
    cleanVisArr();
    visited[start] = 1;
    q.push(start);

    while (!q.empty())
    {
        int curr = q.front();
        q.pop();
        cout << curr << " -> ";

        #pragma omp parallel for
        for (auto &&i : adj[curr])
        {
            if (!visited[i])
            #pragma omp critical
            {
                q.push(i);
                visited[i] = 1;
            }
        }
    }
}

Graph::~Graph()
{
    v = 0;
    adj.clear();
    visited.clear();
}

int main()
{
    int v, e, start;
    cout << "Enter number of vertices and edges : ";
    cin >> v >> e;
    // --------------- Create Graph --------------- //
    Graph g(v, e);
    bool choice;
    cout << "\nUse inBuilt Graph : ";
    cin >> choice;
    if (!choice)
    {
        g.MakeGraph(false);
        cout << "=> Graph created using inputs." << endl;
    }
    else
    {
        g.MakeGraph(true);
        cout << "=> Graph created using inBuilt Graph" << endl;
    }

    // --------------- DFS --------------- //
    cout << "\nEnter starting vertex for DFS : ";
    cin >> start;
    cout << "=> DFS : ";
    g.DFS(start);
    cout <<"(end)"<< endl;

    // --------------- BFS --------------- //
    cout << "\nEnter starting vertex for BFS : ";
    cin >> start;
    cout << "=> BFS : ";
    g.BFS(start);
    cout <<"(end)"<< endl;

    return 0;
}
