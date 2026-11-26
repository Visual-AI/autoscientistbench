// bench-adapt: entry point of one isolated worker; it replaces main.cpp, which drives the paper's sweep and the sealed-bid
// Deng-* mechanisms that need the bids in the clear. eps is the paper's setting from main.cpp.
#include "BFM-SWM.h"

int main(int argc, char* argv[]) {
    string input, output;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (string(argv[i]) == "--input") input = argv[i + 1];
        if (string(argv[i]) == "--output") output = argv[i + 1];
    }
    if (input.empty() || output.empty() || !getenv("BENCH_DATA_DIR")) {
        cerr << "usage: bfm_policy --input FILE --output FILE (with BENCH_DATA_DIR set)" << endl;
        return 2;
    }
    const offer::Instance instance = offer::read_instance(input);
    const string graph_dir = string(getenv("BENCH_DATA_DIR")) + "/" + instance.graph;
    edge_text = graph_dir + "/renum_edge.txt";
    node_text = graph_dir + "/renum_node.txt";
    if (!ifstream(edge_text) || !ifstream(node_text)) {
        cerr << "graph not found: " << graph_dir << endl;
        return 2;
    }
    read_data();

    double eps = 0.1;
    offer::Client oracle;
    const Result result = BFM_SWM(instance.budget, eps, oracle);

    ofstream out(output);
    out.precision(17);
    out << "{\"id\":\"" << instance.id << "\",\"coverage\":" << result.revenue << ",\"payment\":" << result.price
        << ",\"oracle_queries\":" << result.oracle << "}" << endl;
    return out.good() ? 0 : 3;
}
