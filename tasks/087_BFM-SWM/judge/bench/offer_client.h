// C++ client for the private offer service; holds no private state. Python policies use offer_client.py.
// Wire format: 4-byte big-endian length + JSON, see private_wire.py.
#ifndef BENCH_OFFER_CLIENT_H
#define BENCH_OFFER_CLIENT_H
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace offer {

struct Instance {
    std::string id;
    std::string graph;
    double budget = 0.0;
    long algorithm_index = 0;
};

// The worker's --input file: {"id": "...", "graph": "...", "budget": 400.0, "algorithm_index": 3}
inline Instance read_instance(const std::string& path) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    const auto str_field = [&](const char* key) {
        const std::string tag = std::string("\"") + key + "\": \"";
        const size_t a = text.find(tag);
        if (a == std::string::npos) { fprintf(stderr, "input misses %s\n", key); exit(2); }
        const size_t b = a + tag.size();
        return text.substr(b, text.find('"', b) - b);
    };
    const auto num_field = [&](const char* key) {
        const std::string tag = std::string("\"") + key + "\": ";
        const size_t a = text.find(tag);
        if (a == std::string::npos) { fprintf(stderr, "input misses %s\n", key); exit(2); }
        return strtod(text.c_str() + a + tag.size(), nullptr);
    };
    Instance instance;
    instance.id = str_field("id");
    instance.graph = str_field("graph");
    instance.budget = num_field("budget");
    instance.algorithm_index = static_cast<long>(num_field("algorithm_index"));
    return instance;
}

class Client {
public:
    Client() {
        const char* path = getenv("BENCH_RPC_SOCKET");
        if (!path) { fprintf(stderr, "BENCH_RPC_SOCKET is not set\n"); exit(2); }
        fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        strncpy(address.sun_path, path, sizeof(address.sun_path) - 1);
        if (fd_ < 0 || connect(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
            fprintf(stderr, "cannot reach the offer service\n");
            exit(2);
        }
    }
    ~Client() { if (fd_ >= 0) close(fd_); }
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    // Offer prices[i] to ids[i], in order. Returns one '1' (accepted) or '0' (rejected) per offer actually made.
    // With stop_on_accept the offers after the first acceptance are not made and the string is shorter.
    std::string offer(const std::vector<int>& ids, const std::vector<double>& prices, bool stop_on_accept) {
        std::string request = "{\"method\":\"offer\",\"args\":[[";
        request.reserve(ids.size() * 32 + 64);
        char number[64];
        for (size_t i = 0; i < ids.size(); i++) {
            int size = snprintf(number, sizeof(number), i ? ",%d" : "%d", ids[i]);
            request.append(number, size);
        }
        request += "],[";
        for (size_t i = 0; i < prices.size(); i++) {
            int size = snprintf(number, sizeof(number), i ? ",%.17g" : "%.17g", prices[i]);
            request.append(number, size);
        }
        request += stop_on_accept ? "],true],\"kwargs\":{}}" : "],false],\"kwargs\":{}}";
        const std::string reply = roundtrip(request);
        const std::string tag = "\"result\":\"";
        const size_t a = reply.find(tag);
        if (a == std::string::npos) fail(reply);
        const size_t b = a + tag.size();
        return reply.substr(b, reply.find('"', b) - b);
    }

    // Commit the winner set (once).
    void declare(const std::vector<int>& winners) {
        std::string request = "{\"method\":\"declare\",\"args\":[[";
        for (size_t i = 0; i < winners.size(); i++) {
            if (i) request += ",";
            request += std::to_string(winners[i]);
        }
        request += "]],\"kwargs\":{}}";
        const std::string reply = roundtrip(request);
        if (reply.find("\"ok\":true") == std::string::npos) fail(reply);
    }

private:
    int fd_ = -1;

    [[noreturn]] static void fail(const std::string& reply) {
        fprintf(stderr, "offer service refused the request: %s\n", reply.substr(0, 200).c_str());
        exit(3);
    }
    void write_all(const char* data, size_t size) {
        while (size) {
            const ssize_t sent = write(fd_, data, size);
            if (sent <= 0) { fprintf(stderr, "offer service closed\n"); exit(3); }
            data += sent;
            size -= static_cast<size_t>(sent);
        }
    }
    void read_all(char* data, size_t size) {
        while (size) {
            const ssize_t got = read(fd_, data, size);
            if (got <= 0) { fprintf(stderr, "offer service closed\n"); exit(3); }
            data += got;
            size -= static_cast<size_t>(got);
        }
    }
    std::string roundtrip(const std::string& request) {
        const uint32_t out = htonl(static_cast<uint32_t>(request.size()));
        write_all(reinterpret_cast<const char*>(&out), 4);
        write_all(request.data(), request.size());
        uint32_t in = 0;
        read_all(reinterpret_cast<char*>(&in), 4);
        std::string reply(ntohl(in), '\0');
        read_all(&reply[0], reply.size());
        return reply;
    }
};

}  // namespace offer
#endif
