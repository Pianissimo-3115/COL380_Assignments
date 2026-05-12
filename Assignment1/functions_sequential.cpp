#include "functions_sequential.h"

//renaming to avoid collisions

OrderBookEntrySeq decodePacketSeq(uint64_t packet) {
    OrderBookEntrySeq entry;
    entry.stockID = (packet >> 0) & 0xFFFFFFFF;
    entry.orderType = ((packet >> 32) & 0x1) == 1;
    entry.orderQty = (packet >> 33) & 0xFF;
    entry.orderValue = (packet >> 41) & 0xFF;
    
    return entry;
}

uint64_t unstuffBitsSeq(uint64_t packet) {
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
                i+=2; 
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

struct StockInfoSeq {
    uint32_t stockID;
    uint8_t lastSellValue;
    uint8_t lastBuyValue;
    uint8_t minSellValue;
    uint8_t maxBuyValue;
    int64_t sumOrderValue;  
    int orderCount;         
    bool hasSellForMin;     
    bool hasBuyForMax;      
    
    StockInfoSeq() : lastSellValue(0), lastBuyValue(0), 
                  minSellValue(255), maxBuyValue(0),
                  sumOrderValue(0), orderCount(0),
                  hasSellForMin(false), hasBuyForMax(false) {}
    
    int getSpread() const {
        return abs((int)lastSellValue - (int)lastBuyValue);
    }
};

void updateDisplay_seq(const std::vector<uint64_t> &orderBook, int32_t freq) {
    std::map<uint32_t, StockInfoSeq> stockMap;
    int snapCount = 0;
    
    for (size_t i = 0; i < orderBook.size(); i++) {
        uint64_t unstuffed = unstuffBitsSeq(orderBook[i]);
        OrderBookEntrySeq entry = decodePacketSeq(unstuffed);
        
        StockInfoSeq& stock = stockMap[entry.stockID];
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
        
        if ((i + 1) % freq == 0 || i == orderBook.size() - 1) {
            std::vector<StockInfoSeq> stocks;
            for (auto& pair : stockMap) {
                stocks.push_back(pair.second);
            }
            
            std::sort(stocks.begin(), stocks.end(), [](const StockInfoSeq& a, const StockInfoSeq& b) {
                int spreadA = a.getSpread();
                int spreadB = b.getSpread();
                if (spreadA != spreadB) {
                    return spreadA > spreadB;
                }
                return a.stockID > b.stockID;
            });
            
            std::string filename = "snap_correct_" + std::to_string(snapCount) + ".txt";
            std::ofstream outFile(filename);
            
            for (const auto& stock : stocks) {
                int spread = stock.getSpread();
                outFile << stock.stockID << " " 
                        << (int)stock.lastSellValue << " " 
                        << (int)stock.lastBuyValue << " " 
                        << spread << "\n";
            }
            
            outFile.close();
            snapCount++;

            if((orderBook.size()) % freq == 0 && (i + 1) == orderBook.size()) {
                //make a duplicate file
                std::string dupFilename = "snap_correct_" + std::to_string(snapCount) + ".txt";
                std::ofstream dupOutFile(dupFilename);
                for (const auto& stock : stocks) {
                    int spread = stock.getSpread();
                    dupOutFile << stock.stockID << " " 
                            << (int)stock.lastSellValue << " " 
                            << (int)stock.lastBuyValue << " " 
                            << spread << "\n";
                }
                dupOutFile.close();
            }
        }
}
}

int64_t totalAmountTraded_seq(const std::vector<uint64_t> &orderBook) {
    int64_t total = 0;
    
    for (uint64_t packet : orderBook) {
        uint64_t unstuffed = unstuffBitsSeq(packet);
        OrderBookEntrySeq entry = decodePacketSeq(unstuffed);
        
        total += (int64_t)entry.orderQty * (int64_t)entry.orderValue;
    }
    
    return total;
}

void printOrderStats_seq(const std::vector<uint64_t> &orderBook) {
    std::map<uint32_t, StockInfoSeq> stockMap;
    
    for (uint64_t packet : orderBook) {
        uint64_t unstuffed = unstuffBitsSeq(packet);
        OrderBookEntrySeq entry = decodePacketSeq(unstuffed);
        
        StockInfoSeq& stock = stockMap[entry.stockID];
        stock.stockID = entry.stockID;
        
        if (entry.orderType) {
            stock.hasSellForMin = true;
            if (entry.orderValue < stock.minSellValue) {
                stock.minSellValue = entry.orderValue;
            }
        } else {
            stock.hasBuyForMax = true;
            if (entry.orderValue > stock.maxBuyValue) {
                stock.maxBuyValue = entry.orderValue;
            }
        }
        
        stock.sumOrderValue += (int64_t)entry.orderValue;
        stock.orderCount++;
    }
    
    std::ofstream outFile("stats_correct.txt");
    outFile << std::fixed << std::setprecision(4);
    
    for (auto& pair : stockMap) {
        const StockInfoSeq& stock = pair.second;
            
        double avgOrderValue = 0.0;
        if (stock.orderCount > 0) {
            avgOrderValue = static_cast<double>(stock.sumOrderValue) /
                            static_cast<double>(stock.orderCount);
        }
            
        uint8_t minSell = stock.hasSellForMin ? stock.minSellValue : 0;
        uint8_t maxBuy  = stock.hasBuyForMax  ? stock.maxBuyValue  : 0;
    
        outFile << stock.stockID << " "
                << static_cast<int>(minSell) << " "
                << static_cast<int>(maxBuy) << " "
                << std::fixed << std::setprecision(4)
                << avgOrderValue << "\n";
    }
    
    outFile.close();
}
