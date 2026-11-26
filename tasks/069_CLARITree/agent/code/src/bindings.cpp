#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clari_tree.hpp"
#include "clari_tree_const.hpp"

namespace py = pybind11;

namespace {

// Read-only serialization for reproducibility; fitting/prediction are unchanged.
py::dict export_node_state(const Node* node) {
    py::dict out;
    out["is_leaf"] = node->is_leaf;
    out["n_instances"] = node->n_instances;
    if (node->is_leaf) {
        out["coefficients"] = node->coefficients;
    } else {
        out["feature_idx"] = node->feature_idx;
        out["threshold"] = node->threshold;
        out["left"] = export_node_state(node->left);
        out["right"] = export_node_state(node->right);
    }
    return out;
}

template <typename Tree>
void bind_tree_methods(py::class_<Tree>& cls) {
    cls.def("fit",
            [](Tree& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit(X, y, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the tree with (X, y). Returns objective loss.")
       .def("predict",
            [](Tree& self, const Eigen::Ref<const Eigen::MatrixXd>& X) {
                py::gil_scoped_release release;
                return self.predict(X);
            },
            py::arg("X"),
            "Predict values for X.")
       .def("print_tree", &Tree::print_tree)
       .def("get_traversed_thresholds",
            [](const Tree& self) {
                py::dict out;
                const auto all_thresholds = self.get_traversed_thresholds();
                for (std::size_t feature = 1; feature < all_thresholds.size(); ++feature) {
                    py::list thresholds;
                    for (double t : all_thresholds[feature]) {
                        thresholds.append(t);
                    }
                    out[py::int_(feature)] = thresholds;
                }
                return out;
            },
            "Return traversed candidate thresholds by feature index.")
       .def("print_traversed_thresholds", &Tree::print_traversed_thresholds)
       .def("get_threshold_pool",
            [](const Tree& self) {
                py::dict out;
                const auto all_thresholds = self.get_threshold_pool();
                for (std::size_t feature = 1; feature < all_thresholds.size(); ++feature) {
                    py::list thresholds;
                    for (double t : all_thresholds[feature]) {
                        thresholds.append(t);
                    }
                    out[py::int_(feature)] = thresholds;
                }
                return out;
            },
            "Return fit-time global threshold pool by feature index.")
       .def("print_threshold_pool", &Tree::print_threshold_pool)
       .def("n_leaves", &Tree::n_leaves);
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "clari_tree C++ core bindings";

    py::class_<Greedy> greedy(m, "Greedy");
    greedy.def(py::init<double, Depth, double, int, const std::string&, bool, int>(),
               py::arg("kappa"),
               py::arg("depth"),
               py::arg("lambda_") = 0.0,
               py::arg("n_thresholds") = 20,
               py::arg("thresholds_strategy") = "quantile",
               py::arg("verbose") = true,
               py::arg("min_leaf_node_size") = 0)
          .def(py::init<double, Depth, double, int, bool, int>(),
               py::arg("kappa"),
               py::arg("depth"),
               py::arg("lambda_"),
               py::arg("n_thresholds"),
               py::arg("verbose"),
               py::arg("min_leaf_node_size") = 0);
    bind_tree_methods(greedy);
    greedy.def("export_state", [](const Greedy& self) {
        if (self.root == nullptr) throw std::runtime_error("No fitted tree");
        py::dict out;
        out["root"] = export_node_state(self.root);
        out["continuous_idx"] = self.continuous_idx_;
        out["m"] = self.m;
        out["n"] = self.n;
        out["p_reg"] = self.p_reg_;
        out["kappa"] = self.kappa;
        out["lambda"] = self.lambda;
        out["depth"] = self.depth;
        out["n_thresholds"] = self.n_thresholds;
        out["thresholds_strategy"] = self.thresholds_strategy;
        out["min_leaf_node_size"] = self.min_leaf_node_size;
        return out;
    });

    py::class_<CLARITree, Greedy> clari_tree(m, "CLARITree");
    clari_tree.def(py::init<double, Depth, double, int, const std::string&, bool, int>(),
                   py::arg("kappa"),
                   py::arg("depth"),
                   py::arg("lambda_") = 0.0,
                   py::arg("n_thresholds") = 20,
                   py::arg("thresholds_strategy") = "quantile",
                   py::arg("verbose") = true,
                   py::arg("min_leaf_node_size") = 0)
              .def(py::init<double, Depth, double, int, bool, int>(),
                   py::arg("kappa"),
                   py::arg("depth"),
                   py::arg("lambda_"),
                   py::arg("n_thresholds"),
                   py::arg("verbose"),
                   py::arg("min_leaf_node_size") = 0);

    py::class_<GreedyConst> greedy_const(m, "GreedyConst");
    greedy_const.def(py::init<int, double, int, const std::string&, bool, int>(),
                     py::arg("depth"),
                     py::arg("lambda_") = 0.0,
                     py::arg("n_thresholds") = 20,
                     py::arg("thresholds_strategy") = "quantile",
                     py::arg("verbose") = true,
                     py::arg("min_leaf_node_size") = 1)
                .def(py::init<int, double, int, bool, int>(),
                     py::arg("depth"),
                     py::arg("lambda_"),
                     py::arg("n_thresholds"),
                     py::arg("verbose"),
                     py::arg("min_leaf_node_size") = 1);
    bind_tree_methods(greedy_const);

    py::class_<CLARITreeConst, GreedyConst> clari_tree_const(m, "CLARITreeConst");
    clari_tree_const.def(py::init<int, double, int, const std::string&, bool, int>(),
                         py::arg("depth"),
                         py::arg("lambda_") = 0.0,
                         py::arg("n_thresholds") = 20,
                         py::arg("thresholds_strategy") = "quantile",
                         py::arg("verbose") = true,
                         py::arg("min_leaf_node_size") = 1)
                    .def(py::init<int, double, int, bool, int>(),
                         py::arg("depth"),
                         py::arg("lambda_"),
                         py::arg("n_thresholds"),
                         py::arg("verbose"),
                         py::arg("min_leaf_node_size") = 1);
}
