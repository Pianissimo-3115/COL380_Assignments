// TODO:
// remove critical from printOrderStats
// try dynamic schedule



#include "functions.h"
#include <cstdint>
#include <unordered_map>

struct OrderBookEntry {
    uint32_t stockID;
    bool orderType;
    uint8_t orderQty;
    uint8_t orderValue;
};

OrderBookEntry decodePacket(uint64_t packet) {
    OrderBookEntry entry;
    entry.stockID = (packet >> 0) & 0xFFFFFFFF;
    entry.orderType = ((packet >> 32) & 0x1) == 1;
    entry.orderQty = (packet >> 33) & 0xFF;
    entry.orderValue = (packet >> 41) & 0xFF;

    return entry;
}

inline uint64_t unstuffBits(uint64_t packet) {
    uint64_t result = 0;
    int out = 0, ones = 0, i = 0;

    while (out < 49) {
        uint64_t bit = (packet >> i) & 1ULL;
        result |= bit << out;
        out++;

        ones = bit ? ones + 1 : 0;
        i += (ones == 5) ? 2 : 1;
        if (ones == 5) ones = 0;
    }
    return result;
}
struct StockInfo {
    uint32_t stockID;
    uint8_t lastSellValue;
    uint8_t lastBuyValue;
    uint8_t minSellValue;
    uint8_t maxBuyValue;
    int64_t sumOrderValue;
    int orderCount;
    bool hasSellForMin;
    bool hasBuyForMax;

    StockInfo()
        : lastSellValue(0), lastBuyValue(0), minSellValue(255), maxBuyValue(0),
        sumOrderValue(0), orderCount(0), hasSellForMin(false),
        hasBuyForMax(false) {}

    int getSpread() const { return abs((int)lastSellValue - (int)lastBuyValue); }
};


void updateDisplay(const std::vector<uint64_t> &orderBook, int32_t freq) {
    size_t numSnaps = orderBook.size() / freq + 1;

    std::vector<std::unordered_map<uint32_t, StockInfo>> snaps(numSnaps);
    std::vector<std::vector<StockInfo>> allStocks(numSnaps);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < orderBook.size(); i += freq) {
        std::unordered_map<uint32_t, StockInfo> snap;
        for (size_t j = i; j < std::min(orderBook.size(), i + freq); j++) {
            uint64_t unstuffed = unstuffBits(orderBook[j]);
            OrderBookEntry entry = decodePacket(unstuffed);
            StockInfo &stock = snap[entry.stockID];
            stock.stockID = entry.stockID;

            if (entry.orderType) {
                stock.lastSellValue = entry.orderValue;
                stock.hasSellForMin = true;
                if (entry.orderValue < stock.minSellValue)
                    stock.minSellValue = entry.orderValue;
            } else {
                stock.lastBuyValue = entry.orderValue;
                stock.hasBuyForMax = true;
                if (entry.orderValue > stock.maxBuyValue)
                    stock.maxBuyValue = entry.orderValue;
            }
        }
        snaps[i / freq] = std::move(snap);
    }

    std::unordered_map<uint32_t, StockInfo> prev_snap;

    for (size_t k = 0; k < numSnaps; k++) {
        auto &snap = snaps[k];
        auto &stocks = allStocks[k];

        for (auto &[id, stock] : snap) {
            if (!stock.hasSellForMin) {
                stock.lastSellValue = prev_snap[id].lastSellValue;
                stock.hasSellForMin |= prev_snap[id].hasSellForMin;
            }
            if (!stock.hasBuyForMax) {
                stock.lastBuyValue = prev_snap[id].lastBuyValue;
                stock.hasBuyForMax |= prev_snap[id].hasBuyForMax;
            }
            stocks.push_back(stock);
        }

        for (auto &[id, stock] : prev_snap) {
            if (snap.find(id) == snap.end()) {
                stocks.push_back(stock);
                snap.insert({id, stock});
            }
        }

        prev_snap = snap;
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t k = 0; k < numSnaps; k++) {
        auto &stocks = allStocks[k];
        std::sort(stocks.begin(), stocks.end(),
            [](const StockInfo &a, const StockInfo &b) {
                int sa = a.getSpread(), sb = b.getSpread();
                if (sa != sb) return sa > sb;
                return a.stockID > b.stockID;
            });
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t k = 0; k < numSnaps; k++) {
        std::string filename = "snap_" + std::to_string(k) + ".txt";
        FILE *f = fopen(filename.c_str(), "w");

        std::string buffer;
        buffer.reserve(allStocks[k].size() * 32);

        for (const auto &stock : allStocks[k]) {
            char line[64];
            int len = snprintf(
                line, sizeof(line),
                "%u %u %u %d\n",
                stock.stockID,
                (unsigned)stock.lastSellValue,
                (unsigned)stock.lastBuyValue,
                stock.getSpread()
            );
            buffer.append(line, len);
        }

        fwrite(buffer.data(), 1, buffer.size(), f);
        fclose(f);
    }
}

int64_t totalAmountTraded(const std::vector<uint64_t> &orderBook) {
    int64_t total = 0;

    #pragma omp parallel for reduction(+:total)
    for (size_t i = 0; i < orderBook.size(); i++) {
        uint64_t packet = orderBook[i];
        uint64_t unstuffed = unstuffBits(packet);
        OrderBookEntry entry = decodePacket(unstuffed);

        total += (int64_t)entry.orderQty * (int64_t)entry.orderValue;
    }

    return total;
}

void printOrderStats(const std::vector<uint64_t> &orderBook) {
    std::vector<StockInfo> stocks(1e6 + 1);

    #pragma omp parallel
    {
        std::unordered_map<uint32_t, StockInfo> local_stocks;

        #pragma omp for
        for (size_t i = 0; i < orderBook.size(); i++) {
            uint64_t unstuffed = unstuffBits(orderBook[i]);
            OrderBookEntry entry = decodePacket(unstuffed);
            StockInfo &stock = local_stocks[entry.stockID];
            stock.stockID = entry.stockID;

            if (entry.orderType) {
                stock.hasSellForMin = true;
                if (entry.orderValue < stock.minSellValue)
                    stock.minSellValue = entry.orderValue;
            } else {
                stock.hasBuyForMax = true;
                if (entry.orderValue > stock.maxBuyValue)
                    stock.maxBuyValue = entry.orderValue;
            }
            stock.sumOrderValue += entry.orderValue;
            stock.orderCount++;
        }

        #pragma omp critical
        {
            for (const auto &[id, stock] : local_stocks) {
                stocks[id].stockID = id;
                stocks[id].hasSellForMin |= stock.hasSellForMin;
                stocks[id].hasBuyForMax  |= stock.hasBuyForMax;
                stocks[id].minSellValue = std::min(stocks[id].minSellValue, stock.minSellValue);
                stocks[id].maxBuyValue  = std::max(stocks[id].maxBuyValue, stock.maxBuyValue);
                stocks[id].sumOrderValue += stock.sumOrderValue;
                stocks[id].orderCount += stock.orderCount;
            }
        }
    }

    FILE *f = fopen("stats.txt", "w");

    std::string buffer;
    buffer.reserve(1024 * 1024);

    for (const auto &stock : stocks) {
        if (!stock.hasBuyForMax && !stock.hasSellForMin)
            continue;

        double avg = stock.orderCount
            ? (double)stock.sumOrderValue / (double)stock.orderCount
            : 0.0;

        uint8_t minSell = stock.hasSellForMin ? stock.minSellValue : 0;
        uint8_t maxBuy  = stock.hasBuyForMax  ? stock.maxBuyValue  : 0;

        char line[128];
        int len = snprintf(
            line, sizeof(line),
            "%u %u %u %.4f\n",
            stock.stockID,
            (unsigned)minSell,
            (unsigned)maxBuy,
            avg
        );
        buffer.append(line, len);
    }

    fwrite(buffer.data(), 1, buffer.size(), f);
    fclose(f);
}
