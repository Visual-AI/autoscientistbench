
#ifndef COVERAGE_BFM_SWM_H
#define COVERAGE_BFM_SWM_H
#include "Deng-Distorted.h"
#include "offer_client.h"  // bench-adapt

// bench-adapt: every fact about a seller's private cost now arrives as an answer to an offer. The mechanism and its constants
// are the authors'. The decisions that read Groundset[u].cost are marked below; S_cost is reporting only and is now NaN.
Result BFM_SWM(double Budget , double eps, offer::Client& oracle) {

    cout<<"BFM-SWM & Max Budget: "<<Budget<<"-------------------start-------------"<<endl;

    auto start = std::chrono::high_resolution_clock::now();

    int l=1;
    int t=0;
    long long int oracle_times = 0;
    double alpha = 1 + sqrt(6)/2;
    double beta = 3;
    double p_t = eps / alpha;
    S_class A_u = S_class();


    vector<float> pi(node_num,0.0);
    vector<bool> is_active(node_num,false);


    //************** R active users****************
    vector<int> R;

    // bench-adapt: Algorithm 1 line 1 as one batch of offers at the full budget (was: Groundset[u].cost <= Budget).
    vector<int> everyone(node_num);
    for (int u = 0; u < node_num; u++) everyone[u] = u;
    const string first_answers = oracle.offer(everyone, vector<double>(node_num, Budget), false);

    for (int u = 0; u < node_num; u++){

        if(first_answers[u] == '1'){
            is_active[u] = true;
            R.push_back(u);
            pi[u]=Budget;
        }

    }

    vector<S_class> S_prev(l, S_class());
    vector<S_class> S_curr(l,S_class());
    vector<S_class> S_sets;

    S_class A_1 = S_class();
    S_class A_1_temp = S_class();

    // bench-adapt: read-ahead, to avoid one RPC per offer. Until someone accepts, S_curr does not change, so the prices of
    // the next sellers can be computed now. The oracle stops the batch at the first acceptance; later offers are not made.
    vector<signed char> answer(node_num, -1);
    vector<double> asked(node_num, 0.0);
    auto read_ahead = [&](vector<int>::iterator from) {
        vector<int> ids;
        vector<double> prices;
        for (auto jt = from; jt != R.end() && ids.size() < 4096; ++jt) {
            const int v = *jt;
            if (S_prev[0].selected[v] == 1 || A_u.selected[v] == 1) continue;
            float price = pi[v];
            const double lowered = S_curr[0].marginal_gain_g(v) / (beta + p_t/Budget);
            if (lowered < price) price = lowered;
            ids.push_back(v);
            prices.push_back(price);
        }
        const string got = oracle.offer(ids, prices, true);
        for (size_t k = 0; k < got.size(); k++) {
            answer[ids[k]] = got[k] == '1';
            asked[ids[k]] = prices[k];
        }
    };

    bool stop = false;
    while(!stop) {
        A_1 = A_1_temp;


        t = t + 1;
        p_t = alpha * p_t;


        for (int i = 0; i < l; i++) {
            S_curr[i] = S_class();
        }


        for (auto it = R.begin(); it != R.end();) {
            int u = *it;


            bool in_prev = false;
            for (int i = 0; i < l; i++) {
                if (S_prev[i].selected[u]==1||A_u.selected[u]==1) {
                    in_prev = true;
                    break;
                }
            }
            if (in_prev) {
                ++it;
                continue;
            }

            //*********************best_j********************
            int best_j = 0;
            double max_mg = -1;
            for(int j = 0; j < l; j++){
                double mg = S_curr[j].marginal_gain_g(u);
                oracle_times++;
                if(mg > max_mg){
                    max_mg = mg;
                    best_j = j;
                }
            }


            //******************update price*******************
            if(max_mg / (beta + p_t/Budget) < pi[u]){
                pi[u] = max_mg / (beta + p_t/Budget);
            }

            // bench-adapt: was Groundset[u].cost <=  pi[u]
            if (answer[u] < 0) read_ahead(it);
            if (asked[u] != pi[u]) { cerr << "read-ahead price mismatch" << endl; exit(4); }
            const bool accepted = answer[u] == 1;
            answer[u] = -1;
            if(accepted){
                double curr_f = S_curr[best_j].g_S();
                oracle_times++;
                if((curr_f + max_mg) - (S_curr[best_j].S_price + pi[u]) > p_t ){
                    A_u.clear();
                    A_1_temp = S_curr[best_j];
                    A_u.add_element_truth(A_u.marginal_gain_g(u),u,pi[u]);
                    oracle_times++;

                    break;
                }
                else{

                    S_curr[best_j].add_element_truth(max_mg, u,pi[u]);

                    ++it;
                }

            }
            else{

                is_active[u] = false;
                it = R.erase(it);

            }

        }


        stop = true;
        for(int u = 0; u < node_num ;u++){
            if(!is_active[u]) continue;
            bool covered = false;
            for(int i = 0; i < l ;i++){
                if(S_prev[i].selected[u]==1 || S_curr[i].selected[u]==1||A_u.selected[u]==1){
                    covered = true;
                    break;}

            }
            if(!covered){
                stop = false;
                break;
            }
        }

        S_sets.clear();
        for (int i = 0; i < l; i++) {
            S_sets.push_back(S_prev[i]);

            S_sets.push_back(S_curr[i]);

        }

        S_prev = S_curr;

    }
    int M = t;
 //   cout <<"M: "<<M<<endl;

    S_class S_star = S_class();

    double max_v = -1e18;

    for (auto& A : S_sets) {
        double w_S = A.S_revenue - A.S_price ;

        if (w_S > max_v) {
            max_v = w_S;
            S_star = A;
        }
    }
    if(S_star.S_revenue - S_star.S_price < A_u.S_revenue - A_u.S_price){
        cout<<"A_u"<<endl;
        S_star = A_u;
    }


    // bench-adapt: was f_S(), which needs the private costs; the scorer computes v(S) - c(S) from the declared winners.
    S_star.S_revenue = S_star.g_S();
    oracle_times++;


    auto end = std::chrono::high_resolution_clock::now();


    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    long long int total_time_ns = duration.count();


//    cout<<"running time: "<<total_time_ns<<endl;
//    cout<<"S*:"<<endl;
//    cout<<"S*:"<<endl;
//    cout<<"  revenue: "<<S_star.S_revenue << " Price: "<<S_star.S_price<<" Cost: "<<S_star.S_cost<<" size: "<<S_star.solution.size()<<endl;
//    for(const auto &p:S_star.solution)
//        cout<<p<<" ";
//    cout<<endl;
//
//    cout<<"oracle times: "<<oracle_times<<endl;

     // bench-adapt: commit the winners; each one is paid its most recently posted price (Algorithm 1, last line).
     oracle.declare(vector<int>(S_star.solution.begin(), S_star.solution.end()));

     cout<<"coverage: "<<S_star.S_revenue<<endl;
     cout<<"oracle queries: "<<oracle_times<<endl;
     cout<<"BFM-SWM ---------end--------- "<<endl<<endl;

    return Result(S_star.S_revenue,S_star.S_cost,S_star.S_price, S_star.solution.size(),oracle_times,total_time_ns);

}

#endif //COVERAGE_BFM_SWM_H
