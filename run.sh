./build/bin/clang++ -fsanitize=thread -mllvm -tsan-enable-l0=true -O0 -g test.cpp -o test_l0
./build/bin/clang++ -fsanitize=thread -mllvm -tsan-enable-l0=false -O0 -g test.cpp -o test_base
clang++ -isysroot "$(xcrun --show-sdk-path)" -dynamiclib -undefined dynamic_lookup -O0 wrap_tsan.cpp -o libwraptsan.dylib
DYLD_INSERT_LIBRARIES=./libwraptsan.dylib DYLD_FORCE_FLAT_NAMESPACE=1 ./test_base
# r1=50005000 r2=64
# [interpose] reads=10064 writes=10065 total=20129
DYLD_INSERT_LIBRARIES=./libwraptsan.dylib DYLD_FORCE_FLAT_NAMESPACE=1 ./test_l0                
# r1=50005000 r2=64
# [interpose] reads=0 writes=65 total=65