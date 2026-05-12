#include "functions_sequential.h"
#include "functions.h"
#include <assert.h>
#include <random>

int random_int(int l, int r) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(l, r);
    return dist(rng);
}

void printorder(uint64_t packet)
{
    //unstuffBitsSeq
    uint64_t unstuffed = unstuffBitsSeq(packet);
    OrderBookEntrySeq entry = decodePacketSeq(unstuffed);
    std::cout << "StockID: " << entry.stockID << " OrderType: "
    << (entry.orderType == 0 ? "Buy" : "Sell") << " OrderQty: " << (int)entry.orderQty << " OrderValue: " << (int)entry.orderValue << std::endl;
}

void printorderbook(const std::vector<uint64_t> &orderBook)
{
    for(auto packet : orderBook)
    {
        printorder(packet);
    }
}

uint64_t stuff(uint64_t packet)
{
    int consecutiveOnes = 0;
    uint64_t result = 0;
    int outputBitPos = 0;
    for(int i = 0; i < 64 && outputBitPos < 64; ++i)
    {
        bool bit = (packet >> i) & 0x1;
        if(bit)
        {
            consecutiveOnes++;
            result |= (1ULL << outputBitPos);
            outputBitPos++;
            if(consecutiveOnes == 5)
            {
                consecutiveOnes = 0;
                outputBitPos++;
            }
        }
        else
        {
            consecutiveOnes = 0;
            outputBitPos++;
        }
    }
    return result;
}

std::vector<uint64_t> readFromFile(const std::string &filename) {
    std::vector<uint64_t> orderBook;
    std::ifstream inFile(filename, std::ios::binary);

    //reading file line by line 
    uint64_t packet;
    while(inFile.read(reinterpret_cast<char*>(&packet), sizeof(packet))) {
        orderBook.push_back(packet);
        // printorder(packet);
    }
    inFile.close();
    return orderBook;
}

void generateTC(std::string filename, int freq, long long size)
{
    std::vector<uint64_t> orderBook;
    std::ofstream outFile(filename, std::ios::binary);
    for(long long i = 0; i < size; ++i)
    {
        uint32_t stockID = random_int(1, 100);
        bool orderType = random_int(0, 1);
        uint8_t orderQty = random_int(1, 100);
        uint8_t orderValue = random_int(1, 100);
        uint64_t unstuffed = 0;
        unstuffed |= ((uint64_t)stockID) << 0;
        unstuffed |= ((uint64_t)orderType) << 32;
        unstuffed |= ((uint64_t)orderQty) << 33;
        unstuffed |= ((uint64_t)orderValue) << 41;

        uint64_t stuffed = stuff(unstuffed);
        orderBook.push_back(stuffed);
        outFile.write(reinterpret_cast<char*>(&stuffed), sizeof(stuffed));
        // std::cout << "StockID: " << stockID << " OrderType: " << orderType << " OrderQty: " << (int)orderQty << " OrderValue: " << (int)orderValue << std::endl;
    }
    updateDisplay_seq(orderBook, freq);
    printOrderStats_seq(orderBook);
    std::cout << "total amount traded is " << totalAmountTraded_seq(orderBook) << std::endl;
    outFile.close();
}

int main(int argc, char* argv[])
{
    //get the filename, frequency and size from command line arguments
    if(argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <frequency> <size>" << std::endl;
        return 1;
    }
    int freq = std::stoi(argv[1]);
    int size = std::stoll(argv[2]);

    std::string filename = "testcase_freq_" + std::to_string(freq) + "_size_" + std::to_string(size) + ".bin";

    generateTC(filename, freq, size);

    //FOR GENERATING OUTPUT OF PARALLEL CODE
    //uncomment the below lines and generate your outputs

    std::vector<uint64_t> orderBook = readFromFile(filename);
    updateDisplay(orderBook, freq);
    printOrderStats(orderBook);
    std::cout << "total amount traded (parallel version) is " << totalAmountTraded(orderBook) << std::endl;


    //To debug
    // printorderbook(readFromFile(filename));
    return 0;
}
