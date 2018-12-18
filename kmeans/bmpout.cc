#include <iostream>
#include <fstream>
#include <iomanip>

#include "bmpdata.hh"
#include "kmeans.h"

#pragma pack(push, 1)
struct DPIXEL
{
	double x;
	double y;
	double z;
};
#pragma pack(pop)

int main(int argc, char **argv)
{
	std::ifstream f(argv[1], std::ios::binary);
	BITMAPFILEHEADER fh;
	f.read(reinterpret_cast<char *>(&fh), sizeof(fh));

	BITMAPINFOHEADER bh;
	f.read(reinterpret_cast<char *>(&bh), sizeof(bh));

	std::cout << std::setbase(16) << fh.bfType << std::endl;
	std::cout << std::setbase(0);
	std::cout << bh.biCompression << std::endl;
	std::cout << fh.bfSize << ' ' << fh.bOffBits << std::endl;
	std::cout << bh.biWidth << " x " << bh.biHeight << std::endl;

	Header ofh{bh.biWidth * bh.biHeight, 3};

	long skip = (4 - bh.biWidth % 4L) % 4;

	std::ofstream of(argv[2], std::ios::binary);
	of.write(reinterpret_cast<char *>(&ofh), sizeof(ofh));
	f.seekg(fh.bOffBits);
    BITMAPPIXEL px;
    DPIXEL op;
	for (int i = 0; i < bh.biHeight; ++i)
	{
		for (int j = 0; j < bh.biWidth; ++j)
		{
			f.read(reinterpret_cast<char *>(&px), sizeof(px));
            op.x = px.b / 255.;
            op.y = px.g / 255.;
            op.z = px.r / 255.;
			of.write(reinterpret_cast<char *>(&op), sizeof(op));
		}

		f.seekg(skip, std::ios::cur);
	}
}
