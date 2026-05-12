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

uint64_t unstuffBits(uint64_t packet) {
    uint64_t result = 0;
    int outputBitPos = 0;
    int consecutiveOnes = 0;
    const int TARGET_BITS = 49;

    int i = 0;
    while (i < 64 && outputBitPos < TARGET_BITS) {
        bool bit = (packet >> i) & 0x1;

        if (bit) {
            consecutiveOnes++;
            result |= (1ULL << outputBitPos);
            outputBitPos++;

            if (consecutiveOnes == 5) {
                consecutiveOnes = 0;
                i += 2;
                continue;
            }
        } else {
            consecutiveOnes = 0;
            outputBitPos++;
        }
        i++;
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
    int snapCount = 0;
    std::map<size_t, std::unordered_map<uint32_t, StockInfo>> snaps;
    #pragma omp parallel for
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
                if (entry.orderValue < stock.minSellValue) {
                    stock.minSellValue = entry.orderValue;
                }
            } else {
                stock.lastBuyValue = entry.orderValue;
                stock.hasBuyForMax = true;
                if (entry.orderValue > stock.maxBuyValue) {
                    stock.maxBuyValue = entry.orderValue;
                }
            }
        }
        #pragma omp critical
        snaps.insert({i / freq, snap});
    }
    std::unordered_map<uint32_t, StockInfo> prev_snap;
    for (auto &[_, snap] : snaps) {
        std::vector<StockInfo> stocks;
        for (auto &[id, stock] : snap) {
            if(!stock.hasSellForMin) stock.lastSellValue = prev_snap[id].lastSellValue;
            if(!stock.hasBuyForMax) stock.lastBuyValue = prev_snap[id].lastBuyValue;
            stocks.push_back(stock);
        }
        for (auto &[id, stock] : prev_snap){
            if(snap.find(id) == snap.end()){
                stocks.push_back(stock);
                snap.insert({id, stock});
            }
        }
        std::sort(stocks.begin(), stocks.end(),
                  [](const StockInfo &a, const StockInfo &b) {
                  int spreadA = a.getSpread();
                  int spreadB = b.getSpread();
                  if (spreadA != spreadB) {
                  return spreadA > spreadB;
                  }
                  return a.stockID > b.stockID;
                  });
        std::string filename = "snap_" + std::to_string(snapCount) + ".txt";
        std::ofstream outFile(filename);

        for (const auto &stock : stocks) {
            int spread = stock.getSpread();
            outFile << stock.stockID << " " << (int)stock.lastSellValue << " "
                << (int)stock.lastBuyValue << " " << spread << "\n";
        }

        outFile.close();
        snapCount++;
        prev_snap = snap;
    }
    if (orderBook.size() % freq == 0) {
        std::vector<StockInfo> stocks;
        for (auto &pair : prev_snap) {
            stocks.push_back(pair.second);
        }
        std::sort(stocks.begin(), stocks.end(),
                  [](const StockInfo &a, const StockInfo &b) {
                  int spreadA = a.getSpread();
                  int spreadB = b.getSpread();
                  if (spreadA != spreadB) {
                  return spreadA > spreadB;
                  }
                  return a.stockID > b.stockID;
                  });
        std::string filename = "snap_" + std::to_string(snapCount) + ".txt";
        std::ofstream outFile(filename);

        for (const auto &stock : stocks) {
            int spread = stock.getSpread();
            outFile << stock.stockID << " " << (int)stock.lastSellValue << " "
                << (int)stock.lastBuyValue << " " << spread << "\n";
        }

        outFile.close();
        snapCount++;
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
    std::map<uint32_t, StockInfo> stocks;
    #pragma omp parallel
    {
        std::unordered_map<uint32_t, StockInfo> local_stocks;
        #pragma omp for
        for(size_t i = 0; i < orderBook.size(); i++){
            uint64_t unstuffed = unstuffBits(orderBook[i]);
            OrderBookEntry entry = decodePacket(unstuffed);
            StockInfo& stock = local_stocks[entry.stockID];
            stock.stockID = entry.stockID;

            if(entry.orderType) {
                stock.hasSellForMin = true;
                if(entry.orderValue < stock.minSellValue){
                    stock.minSellValue = entry.orderValue;
                }
            }
            else {
                stock.hasBuyForMax = true;
                if(entry.orderValue > stock.maxBuyValue){
                    stock.maxBuyValue = entry.orderValue;
                }
            }
            stock.sumOrderValue += (int64_t)entry.orderValue;
            stock.orderCount++;
        }

        #pragma omp critical
        {   
            for(const auto& [id, stock] : local_stocks){
                stocks[id].stockID = id;
                stocks[id].hasSellForMin |= stock.hasSellForMin;
                stocks[id].hasBuyForMax |= stock.hasBuyForMax;
                stocks[id].minSellValue = std::min(stocks[id].minSellValue, stock.minSellValue);
                stocks[id].maxBuyValue = std::max(stocks[id].maxBuyValue, stock.maxBuyValue);
                stocks[id].sumOrderValue += stock.sumOrderValue;
                stocks[id].orderCount += stock.orderCount;
            }
        }
    }
    std::ofstream outFile("stats.txt");
    outFile << std::fixed << std::setprecision(4);
    for(const auto& [id, stock] : stocks){

        double avgOrderValue = 0.0;
        if (stock.orderCount > 0){
            avgOrderValue = static_cast<double>(stock.sumOrderValue)/ static_cast<double>(stock.orderCount);
        }
        uint8_t minSell = stock.hasSellForMin ? stock.minSellValue : 0;
        uint8_t maxBuy  = stock.hasBuyForMax  ? stock.maxBuyValue  : 0;
    
        outFile << stock.stockID << " "
                << static_cast<int>(minSell) << " "
                << static_cast<int>(maxBuy) << " "
                << std::fixed << std::setprecision(3)
                << avgOrderValue << "\n";
    }
    outFile.close();

}
