#include <bits/stdc++.h>
using namespace std;

int N, E, B;
vector<int> p, c;
vector<vector<uint8_t>> adj;
int P_max = 0;
vector<int> best_clique;
vector<int> cur_clique;

// Reusable buffers to avoid repeated allocations
vector<vector<int>> g_color_classes;
int g_prev_num_colors = 0;
vector<int> g_color_buf;

int GreedyColorBound(vector<int> &cand)
{
    if (cand.empty())
        return 0;

    // Sort candidates by profit descending for greedy coloring
    g_color_buf.assign(cand.begin(), cand.end());
    sort(g_color_buf.begin(), g_color_buf.end(), [](int a, int b)
         { return p[a] > p[b]; });

    // Clear previously used color classes (size->0, capacity kept)
    for (int k = 0; k < g_prev_num_colors; ++k)
        g_color_classes[k].clear();

    int num_colors = 0;
    int U = 0;
    for (int i = 0; i < (int)g_color_buf.size(); ++i)
    {
        int v = g_color_buf[i];
        bool assigned = false;
        for (int k = 0; k < num_colors; ++k)
        {
            auto &cls = g_color_classes[k];
            bool can_assign = true;
            for (int u : cls)
            {
                if (adj[v][u])
                {
                    can_assign = false;
                    break;
                }
            }
            if (can_assign)
            {
                cls.push_back(v);
                assigned = true;
                break;
            }
        }
        if (!assigned)
        {
            if (num_colors < (int)g_color_classes.size())
            {
                g_color_classes[num_colors].push_back(v);
            }
            else
            {
                g_color_classes.push_back({v});
            }
            num_colors++;
            U += p[v];
        }
    }
    g_prev_num_colors = num_colors;
    return U;
}

double KnapsackBound(const vector<int> &cand, int rem)
{
    if (rem <= 0 || cand.empty())
        return 0.0;

    // cand is already sorted by p/c ratio descending
    double bound = 0.0;
    int bound_int = 0;
    for (int v : cand)
    {
        if (c[v] <= rem)
        {
            bound_int += p[v];
            rem -= c[v];
        }
        else
        {
            bound += (double)p[v] * rem / c[v];
            break;
        }
    }
    return bound + bound_int;
}

void FindClique(vector<int> &cand, int P_curr, int W_curr)
{
    int U_color = GreedyColorBound(cand);
    if (P_curr + U_color <= P_max)
        return;

    double U_knap = KnapsackBound(cand, B - W_curr);
    if (P_curr + U_knap <= P_max)
        return;

    for(int i = 0; i < (int)cand.size(); ++i)
    {
        int v = cand[i];

        if (W_curr + c[v] > B)
            continue;

        cur_clique.push_back(v);
        int new_profit = P_curr + p[v];
        int new_weight = W_curr + c[v];

        if (new_profit > P_max)
        {
            P_max = new_profit;
            best_clique = cur_clique;
        }

        vector<int> C_next;
        C_next.reserve(cand.size());

        for (int j = i + 1; j < (int)cand.size(); ++j)
        {
            if (adj[v][cand[j]])
                C_next.push_back(cand[j]);
        }

        if (!C_next.empty())
        {
            FindClique(C_next, new_profit, new_weight);
        }

        cur_clique.pop_back();
    }
}
int main(int argc, char *argv[])
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3)
        return 1;

    ifstream fin(argv[1]);
    if (!fin)
        return 1;

    fin >> N >> E >> B;
    adj.assign(N, vector<uint8_t>(N, 0));
    p.reserve(N);
    c.reserve(N);

    int pi, ci;
    for (int i = 0; i < N; i++)
    {
        fin >> pi >> ci;
        p.push_back(pi);
        c.push_back(ci);
    }
    for (int i = 0; i < E; i++)
    {
        int u, v;
        fin >> u >> v;
        adj[u][v] = adj[v][u] = 1;
    }
    fin.close();

    vector<int> C_cand;
    C_cand.reserve(N);
    for (int i = 0; i < N; ++i)
    {
        if (c[i] <= B)
            C_cand.push_back(i);
    }

    sort(C_cand.begin(), C_cand.end(), [](int a, int b)
         { return 1LL * p[a] * c[b] > 1LL * p[b] * c[a]; });

    

    FindClique(C_cand, 0, 0);

    vector<int> result = best_clique;
    sort(result.begin(), result.end());

    ofstream fout(argv[2]);
    if (!fout)
        return 1;
    fout << P_max << "\n";
    for (int i = 0; i < (int)result.size(); ++i)
    {
        if (i)
            fout << ' ';
        fout << result[i];
    }
    fout << "\n";
    fout.close();

    return 0;
}
