#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;
#define SYNC_THRESHOLD 500

int N, E, B;
vector<int> p, c;
vector<vector<uint8_t>> adj;

int P_max = 0;
int my_best_profit = 0;
vector<int> best_clique;
vector<int> cur_clique;

vector<vector<int>> g_color_classes;
int g_prev_num_colors = 0;
vector<int> g_color_buf;

MPI_Win win;
int sync_counter = 0;
bool use_rma = false

int g_nprocs    = 1;
int g_num_tasks = 0;
vector<int> local_spare_buf; 
bool g_should_split = false;

struct Task
{
    vector<int> cand;
    vector<int> clique;
    int P_curr, W_curr;
};
vector<Task> all_tasks;

static void SerializeTask(const Task &t, vector<int> &buf)
{
    buf.push_back((int)t.cand.size());
    buf.insert(buf.end(), t.cand.begin(), t.cand.end());
    buf.push_back((int)t.clique.size());
    buf.insert(buf.end(), t.clique.begin(), t.clique.end());
    buf.push_back(t.P_curr);
    buf.push_back(t.W_curr);
}

static vector<Task> DeserializeTasks(const vector<int> &buf)
{
    vector<Task> tasks;
    int pos = 0;
    while (pos < (int)buf.size())
    {
        Task t;
        int cs = buf[pos++];
        t.cand.assign(buf.begin() + pos, buf.begin() + pos + cs);
        pos += cs;
        int cls = buf[pos++];
        t.clique.assign(buf.begin() + pos, buf.begin() + pos + cls);
        pos += cls;
        t.P_curr = buf[pos++];
        t.W_curr = buf[pos++];
        tasks.push_back(move(t));
    }
    return tasks;
}

int GreedyColorBound(vector<int> &cand)
{
    if (cand.empty())
        return 0;

    g_color_buf.assign(cand.begin(), cand.end());
    sort(g_color_buf.begin(), g_color_buf.end(), [](int a, int b)
         { return p[a] > p[b]; });

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
                g_color_classes[num_colors].push_back(v);
            else
                g_color_classes.push_back({v});

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
    if (use_rma && (++sync_counter % SYNC_THRESHOLD == 0))
    {
        
        int gp;
        MPI_Fetch_and_op(&my_best_profit, &gp, MPI_INT, 0, (MPI_Aint)1,
                         MPI_MAX, win);
        MPI_Win_flush(0, win);
        if (gp > P_max)
            P_max = gp;

        if (!g_should_split)
        {
            int zero = 0, cur_ctr;
            MPI_Fetch_and_op(&zero, &cur_ctr, MPI_INT, 0, (MPI_Aint)0,
                             MPI_SUM, win);
            MPI_Win_flush(0, win);
            
            if (g_num_tasks - cur_ctr < 2 * g_nprocs)
                g_should_split = true;
        }
    }

    int U_color = GreedyColorBound(cand);
    if (P_curr + U_color <= P_max)
        return;

    double U_knap = KnapsackBound(cand, B - W_curr);
    if (P_curr + U_knap <= P_max)
        return;

    for (int i = 0; i < (int)cand.size(); ++i)
    {
        int v = cand[i];

        if (W_curr + c[v] > B)
            continue;

        cur_clique.push_back(v);
        int new_profit = P_curr + p[v];
        int new_weight = W_curr + c[v];

        if (new_profit > P_max)
            P_max = new_profit;
        if (new_profit > my_best_profit)
        {
            my_best_profit = new_profit;
            best_clique    = cur_clique;
        }

        vector<int> C_next;
        C_next.reserve(cand.size());
        for (int j = i + 1; j < (int)cand.size(); ++j)
            if (adj[v][cand[j]])
                C_next.push_back(cand[j]);

        if (!C_next.empty())
        {
            if (g_should_split)
            {
                
                Task spare;
                spare.cand   = move(C_next);
                spare.clique = cur_clique;  
                spare.P_curr = new_profit;
                spare.W_curr = new_weight;
                SerializeTask(spare, local_spare_buf);
            }
            else
            {
                FindClique(C_next, new_profit, new_weight);
            }
        }

        cur_clique.pop_back();
    }
}

void GenerateTasks(vector<int> &cand, int P_curr, int W_curr, int depth)
{
    int U_color = GreedyColorBound(cand);
    if (P_curr + U_color <= P_max)
        return;
    double U_knap = KnapsackBound(cand, B - W_curr);
    if (P_curr + U_knap <= P_max)
        return;

    for (int i = 0; i < (int)cand.size(); ++i)
    {
        int v = cand[i];
        if (W_curr + c[v] > B)
            continue;

        cur_clique.push_back(v);
        int np = P_curr + p[v];
        int nw = W_curr + c[v];

        if (np > P_max)
            P_max = np;
        if (np > my_best_profit)
        {
            my_best_profit = np;
            best_clique    = cur_clique;
        }

        vector<int> C_next;
        C_next.reserve(cand.size());
        for (int j = i + 1; j < (int)cand.size(); ++j)
            if (adj[v][cand[j]])
                C_next.push_back(cand[j]);

        if (!C_next.empty())
        {
            if (depth <= 1)
                all_tasks.push_back({move(C_next), cur_clique, np, nw});
            else
                GenerateTasks(C_next, np, nw, depth - 1);
        }

        cur_clique.pop_back();
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3)
    {
        MPI_Finalize();
        return 1;
    }

    
    ifstream fin(argv[1]);
    if (!fin)
    {
        MPI_Finalize();
        return 1;
    }
    fin >> N >> E >> B;
    adj.assign(N, vector<uint8_t>(N, 0));
    p.resize(N);
    c.resize(N);
    for (int i = 0; i < N; i++)
        fin >> p[i] >> c[i];
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
        if (c[i] <= B)
            C_cand.push_back(i);
    sort(C_cand.begin(), C_cand.end(), [](int a, int b)
         { return 1LL * p[a] * c[b] > 1LL * p[b] * c[a]; });

    
    if (nprocs == 1)
    {
        FindClique(C_cand, 0, 0);
        sort(best_clique.begin(), best_clique.end());
        ofstream fout(argv[2]);
        fout << my_best_profit << "\n";
        for (int i = 0; i < (int)best_clique.size(); i++)
        {
            if (i)
                fout << ' ';
            fout << best_clique[i];
        }
        fout << "\n";
        MPI_Finalize();
        return 0;
    }

    

    
    int num_tasks = 0;
    if (rank == 0)
    {
        int depth = ((int)C_cand.size() >= 8 * nprocs) ? 1 : 2;
        GenerateTasks(C_cand, 0, 0, depth);

        if ((int)all_tasks.size() < 4 * nprocs)
        {
            vector<Task> prev = move(all_tasks);
            all_tasks.clear();
            for (auto &t : prev)
            {
                if ((int)t.cand.size() > 1)
                {
                    cur_clique = t.clique;
                    GenerateTasks(t.cand, t.P_curr, t.W_curr, 1);
                }
                else
                {
                    all_tasks.push_back(move(t));
                }
            }
        }

        num_tasks = (int)all_tasks.size();
    }

    
    MPI_Bcast(&P_max, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        my_best_profit = 0;

    MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (num_tasks > 0)
    {
        
        {
            vector<int> task_buf;
            if (rank == 0)
                for (auto &t : all_tasks)
                    SerializeTask(t, task_buf);

            int buf_size = (int)task_buf.size();
            MPI_Bcast(&buf_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0)
                task_buf.resize(buf_size);
            MPI_Bcast(task_buf.data(), buf_size, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank != 0)
                all_tasks = DeserializeTasks(task_buf);
        }

        
        int *win_mem = nullptr;
        MPI_Aint win_size = (rank == 0) ? (MPI_Aint)(2 * sizeof(int)) : 0;
        MPI_Win_allocate(win_size, (int)sizeof(int), MPI_INFO_NULL,
                         MPI_COMM_WORLD, &win_mem, &win);
        if (rank == 0)
        {
            win_mem[0] = 0;      
            win_mem[1] = P_max;  
        }
        MPI_Barrier(MPI_COMM_WORLD);

        
        MPI_Win_lock_all(0, win);
        use_rma     = true;
        g_nprocs    = nprocs;
        g_num_tasks = num_tasks;

        int one = 1;

        for (;;)
        {
            
            for (;;)
            {
                
                int task_idx;
                MPI_Fetch_and_op(&one, &task_idx, MPI_INT, 0, (MPI_Aint)0,
                                 MPI_SUM, win);
                MPI_Win_flush(0, win);
                if (task_idx >= g_num_tasks)
                    break;

                
                {
                    int gp;
                    MPI_Fetch_and_op(&my_best_profit, &gp, MPI_INT, 0,
                                     (MPI_Aint)1, MPI_MAX, win);
                    MPI_Win_flush(0, win);
                    if (gp > P_max)
                        P_max = gp;
                }

                g_should_split = false;
                cur_clique     = all_tasks[task_idx].clique;
                sync_counter   = 0;
                FindClique(all_tasks[task_idx].cand,
                           all_tasks[task_idx].P_curr,
                           all_tasks[task_idx].W_curr);

                
                {
                    int gp;
                    MPI_Fetch_and_op(&my_best_profit, &gp, MPI_INT, 0,
                                     (MPI_Aint)1, MPI_MAX, win);
                    MPI_Win_flush(0, win);
                    if (gp > P_max)
                        P_max = gp;
                }
            }
            

            use_rma = false;
            MPI_Win_unlock_all(win);

            
            {
                int best_sync;
                MPI_Allreduce(&my_best_profit, &best_sync, 1, MPI_INT,
                              MPI_MAX, MPI_COMM_WORLD);
                if (best_sync > P_max)
                    P_max = best_sync;
            }

            
            int local_sz = (int)local_spare_buf.size();

            vector<int> all_sizes(nprocs);
            MPI_Allgather(&local_sz, 1, MPI_INT,
                          all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

            int total_sz = 0;
            for (int s : all_sizes)
                total_sz += s;

            
            if (total_sz == 0)
                break;

            vector<int> displs(nprocs, 0);
            for (int i = 1; i < nprocs; i++)
                displs[i] = displs[i - 1] + all_sizes[i - 1];

            vector<int> global_buf(total_sz);
            MPI_Allgatherv(local_spare_buf.data(), local_sz, MPI_INT,
                           global_buf.data(), all_sizes.data(), displs.data(),
                           MPI_INT, MPI_COMM_WORLD);

            local_spare_buf.clear();

            
            all_tasks   = DeserializeTasks(global_buf);
            g_num_tasks = (int)all_tasks.size();

            
            if (rank == 0)
            {
                win_mem[0] = 0;
                win_mem[1] = P_max;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Win_lock_all(0, win);
            use_rma = true;
        }
        

        MPI_Win_free(&win);
    }

    
    struct { int val; int idx; } local_in = {my_best_profit, rank}, global_out;
    MPI_Allreduce(&local_in, &global_out, 1, MPI_2INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);

    int winner    = global_out.idx;
    int clique_sz = (rank == winner) ? (int)best_clique.size() : 0;
    MPI_Bcast(&clique_sz, 1, MPI_INT, winner, MPI_COMM_WORLD);
    if (rank != winner)
        best_clique.resize(clique_sz);
    if (clique_sz > 0)
        MPI_Bcast(best_clique.data(), clique_sz, MPI_INT, winner,
                  MPI_COMM_WORLD);

    
    if (rank == 0)
    {
        sort(best_clique.begin(), best_clique.end());
        ofstream fout(argv[2]);
        fout << global_out.val << "\n";
        for (int i = 0; i < (int)best_clique.size(); i++)
        {
            if (i)
                fout << ' ';
            fout << best_clique[i];
        }
        fout << "\n";
    }

    MPI_Finalize();
    return 0;
}