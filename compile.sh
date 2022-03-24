cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DGALAX_LINK_SDL2=ON -DGALAX_LINK_OMP=ON -DCMAKE_CXX_FLAGS="-mavx2" ..
cmake --build "build" --config Release --parallel
