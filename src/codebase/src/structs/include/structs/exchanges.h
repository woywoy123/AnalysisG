#ifndef STRUCTS_EXCHANGES_H
#define STRUCTS_EXCHANGES_H
#include <iostream>
#include <string>

struct ex_error_t {
    std::string message = ""; 
    std::string time = ""; 
}; 

struct exchange_t {
    public:
        int auth_index; 
        std::string auth_header = "CORE-EXCHANGE"; 
        std::string name = "core"; 

        // hdf5 attributes  
        // options: "read", "write"
        std::string cache_mode = "write";

        // optional: specifies the dataset name
        std::string cache_time = "null"; 

        int rate_limit = -1; //requests per second
        std::string domain      = ""; 
        std::string exchange    = ""; 
        std::string replicators = ""; 
        std::string assets      = ""; 
        std::string markets     = ""; 
        std::string orderbook   = ""; 
        std::string deposits    = ""; 
        std::string withdraws   = ""; 

        std::string public_websocket  = ""; 
        std::string private_websocket = ""; 

        std::string user        = ""; 
        std::string wallet      = ""; 
        std::string balance     = "";
        std::string orders      = ""; 

        // cache paths
        std::string coin_cache   = "cache_coins"; 
        std::string market_cache = "cache_market";
        std::string book_cache   = "cache_book"; 
}; 

struct eth_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "ETHEREUM"; 
    std::string name = "ETH"; 
}; 

struct binance_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "BINANCE-EXCHANGE"; 
    std::string name = "Binance"; 

    // requests
    std::string domain    = "https://api.binance.com"; 
    std::string assets    = "https://api.binance.com/sapi/v1/margin/allAssets";
    std::string orderbook = "https://api.binance.com/api/v3/depth";

    // websocket 
    std::string private_websocket = "wss://ws-api.binance.com:9443/ws-api/v3"; 
    std::string public_websocket  = "wss://stream.binance.com:9443/ws";
    
    // posts
    //std::string user    = "https://api.binance.com"; 
    std::string wallet    = "https://api.binance.com/sapi/v1/captial/deposit/address/list"; 
    std::string balance   = "https://api.binance.com/sapi/v1/capital/config/getall";
    std::string trades    = "https://api.binance.com/sapi/v3";
    std::string orders     = "https://api.binance.com/api/v3/order"; 

}; 

struct kraken_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "KRAKEN-EXCHANGE"; 
    std::string name = "Kraken"; 

    // requests
    std::string domain      = "https://api.kraken.com";
    std::string orderbook   = "/0/public/Depth"; 

    std::string assets      = "/0/public/Assets"; 
    std::string markets     = "/0/public/AssetPairs"; 
    std::string deposits    = "/0/private/DepositAddresses"; 
    std::string withdraws   = "/0/private/Withdraw"; 

    // websocket
    std::string ws_auth   = "/0/private/GetWebSocketsToken";
    std::string public_websocket  = "wss://ws.kraken.com/v2";
    std::string private_websocket = "wss://ws-auth.kraken.com/v2";

    // unused
    std::string ping = ""; 

    // posts
    std::string user    = ""; 
    std::string wallet  = "/0/private/DepositAddresses"; 
    std::string balance = "/0/private/Balance";
    std::string ledgers = "/0/private/QueryLedgers";
    std::string orders  = "/0/private/AddOrder"; 
    std::string cancel  = "/0/private/CancelOrder";
}; 

struct okx_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "OKX-EXCHANGE"; 
    std::string name = "OKX"; 

    std::string domain    = "https://www.okx.com";
    std::string orderbook = "/api/v5/market/books-full";
    std::string assets    = "/api/v5/asset/currencies"; 
    std::string wallet    = "/api/v5/asset/deposit-address"; 
    std::string balance   = "/api/v5/account/balance"; 
    std::string markets   = "/api/v5/market/tickers";
    std::string orders    = "/api/v5/trade/order";
    std::string cancel    = "/api/v5/trade/cancel-order";
    std::string trade_fee = "/api/v5/account/trade-fee"; 

    std::string public_websocket  = "wss://ws.okx.com:8443/ws/v5/public";
    std::string private_websocket = "wss://ws.okx.com:8443/ws/v5/private";

}; 

struct bitfinex_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "BITFINEX-EXCHANGE"; 
    std::string name = "Bitfinex"; 

    // requests 
    std::string domain    = "https://api.bitfinex.com/v2/";
    std::string assets    = "conf/"; 
    std::string markets   = "conf/pub:info:pair"; 
    std::string orderbook = "book/"; 
    std::string deposits  = ""; 

    // websocket
    std::string public_websocket  = "wss://api-pub.bitfinex.com/ws/2";
    std::string private_websocket = "wss://api.bitfinex.com/ws/2";

    // posts
    std::string user      = "auth/r/permissions"; 
    std::string wallet    = "auth/r/wallets"; 
    std::string withdraws = "auth/w/withdraw"; 
    std::string balance   = "auth/calc/order/avail";
    std::string orders    = "auth/w/order/submit"; 
    std::string trade_fee = "auth/r/summary"; 
}; 












struct idex_t : public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "IDEX-EXCHANGE"; 
    std::string name = "IDEX"; 
  
    // requests per second 
    int rate_limit = 5;

    // requests
    std::string domain      = "https://api-matic.idex.io"; 
    std::string time        = "https://api-matic.idex.io/v1/time"; 
    std::string exchange    = "https://api-matic.idex.io/v1/exchange"; 
    std::string replicators = "https://sc.idex.io/replicators"; 
    std::string assets      = "https://api-matic.idex.io/v1/assets"; 
    std::string markets     = "https://api-matic.idex.io/v1/markets"; 
    std::string orderbook   = "https://api-matic.idex.io/v1/orderbook"; 
    std::string deposits    = "https://api-matic.idex.io/v1/deposits"; 
    std::string withdraws   = "https://api-matic.idex.io/v1/withdrawals"; 

    // websocket
    std::string ws_entry    = "wss://websocket-matic.idex.io/v1";

    // unused
    std::string ping = "https://api-matic.idex.io/v1/ping"; 

    // posts
    std::string user    = "https://api-matic.idex.io/v1/user"; 
    std::string wallet  = "https://api-matic.idex.io/v1/wallets"; 
    std::string balance = "https://api-matic.idex.io/v1/balances";
    std::string orders  = "https://api-matic.idex.io/v1/orders"; 

}; 

struct kucoin_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "KUCOIN-EXCHANGE"; 
    std::string name = "Kucoin"; 

    // requests per second 
    int rate_limit = 5;

    // requests
    std::string domain      = "https://api.kucoin.com";
    std::string time        = "https://api.kucoin.com/api/v1/timestamp"; 
    std::string exchange    = ""; 
    std::string assets      = "https://api.kucoin.com/api/v3/currencies"; 
    std::string markets     = "https://api.kucoin.com/api/v2/symbols"; 
    std::string orderbook   = "/api/v3/market/orderbook/level2"; 
    std::string deposits    = "/api/v1/deposits"; 
    std::string withdraws   = "/api/v1/withdrawals"; 

    // websocket
    std::string pub_token   = "/api/v1/bullet-public";
    std::string pri_token   = "/api/v1/bullet-private";
    std::string ws_entry    = "wss://ws-api-spot.kucoin.com";

    // unused
    std::string ping = ""; 

    // posts
    std::string user    = "/api/v2/user-info"; 
    std::string wallet  = "/api/v1/deposit-addresses"; 
    std::string balance = "/api/v1/accounts";
    std::string ledgers = "/api/v1/accounts/ledgers";
    std::string orders  = "/api/v1/orders"; 
}; 

struct nexo_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "NEXO-EXCHANGE"; 
    std::string name = "NEXO"; 

    // requests
    std::string domain  = "https://pro-api.nexo.io"; 
    std::string balance = "/api/v2/accountSummary"; 
    std::string user    = "/api/v1/feeTier"; 
    std::string markets = "/api/v1/pairs"; 
    std::string orders  = "/api/v1/orders"; 
}; 




struct bybit_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "BYBIT-EXCHANGE"; 
    std::string name = "ByBit"; 

    // // requests per second
    int rate_limit = 5;

    // requests
    std::string domain    = "https://api.bybit.com"; 
    // std::string domain = "https://api-testnet.bybit.com";
    std::string time      = "/v5/market/time"; 
    // std::string assets    = "/v5/asset/transfer/query-asset-info";
    std::string assets    = "/v5/asset/coin/query-info";
    std::string markets   = "/v5/market/instruments-info"; 
    std::string orderbook = "/v5/market/orderbook";
    std::string balance   = "/v5/account/wallet-balance"; 
    std::string trades    = "/v5/position/list";

    // websocket 
    std::string ws_entry  = "wss://stream.bybit.com/v5/public/spot";
    // std::string ws_entry  = "wss://stream.bybit.com/v5/private";
    
    // posts
    std::string orders    = "/v5/order/create"; 
    std::string cancel    = "/v5/order/cancel";
    std::string deposits  = "/v5/asset/deposit/deposit-to-account"; 
    std::string withdraws = "/v5/asset/withdraw/create"; 
}; 

struct gateio_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "GATEIO-EXCHANGE"; 
    std::string name = "GateIO"; 
}; 

struct bitmex_t: public exchange_t {
    int auth_index = 3; 
    std::string auth_header = "BITMEX-EXCHANGE"; 
    std::string name = "BitMex"; 
}; 




#endif
