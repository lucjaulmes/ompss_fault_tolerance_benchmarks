#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include "bmpdata.hh"
#include "kmeans.h"

#pragma pack(push, 1)
typedef double DPIXEL[3];
#pragma pack(pop)


static inline
uint8_t clip_8b(double d)
{
	if (d < 0.)
		return 0;
	else if (d >= 255.)
		return 255;
	else
		return uint8_t(d);
}


int main(int argc, char **argv)
{
	if (argc < 3 || argc > 4)
	{
		std::cerr << "Usage: " << argv[0] << " in.bmp points.out\nOr:    " << argv[0] << " in.bmp averaged_points.in out.bmp\n";
		return 1;
	}

	std::ifstream bmp_ifs(argv[1], std::ios::binary);
	BITMAPFILEHEADER fh;
	BITMAPINFOHEADER bh;
	bmp_ifs.read(reinterpret_cast<char *>(&fh), sizeof(fh));
	bmp_ifs.read(reinterpret_cast<char *>(&bh), sizeof(bh));

	std::cout << "Converting bitmap " << argv[1] << " with header information:"
		<< "\n  file type=0x" << std::setbase(16) << fh.bfType << std::setbase(0)
		<< "\n  compression=" << bh.biCompression
		<< "\n  file size=" << fh.bfSize
		<< "\n  data offset=" << fh.bOffBits
		<< "\n  dimensions=" << bh.biWidth << " x " << bh.biHeight << '\n';

	// rows of bits are aligned to 4B, skip at the end of every row
	const long skip = (4 - bh.biWidth % 4L) % 4;

	if (argc == 3)
	{
		std::cout << "Converting to input file for kmeans " << argv[2] << '\n';
		Header ofh = {bh.biWidth * bh.biHeight, 3};

		std::ofstream kmeans_in_ofs(argv[2], std::ios::binary);
		kmeans_in_ofs.write(reinterpret_cast<char *>(&ofh), sizeof(ofh));

		bmp_ifs.seekg(fh.bOffBits);
		for (int i = 0; i < bh.biHeight; ++i)
		{
			for (int j = 0; j < bh.biWidth; ++j)
			{
				BITMAPPIXEL p;
				bmp_ifs.read(reinterpret_cast<char *>(&p), sizeof(p));

				DPIXEL op = {p.b / 255., p.g / 255., p.r / 255.};
				kmeans_in_ofs.write(reinterpret_cast<char *>(&op), sizeof(op));
			}

			bmp_ifs.seekg(skip, std::ios::cur);
		}
	}
	else if (argc == 4)
	{
		std::cout << "Converting using output file from kmeans " << argv[2]
					<< " into a bmp with averaged colors " << argv[3] << '\n';

		// start by loading & converting centres.
		std::ifstream means_ifs(argv[2], std::ios::binary);
		RHeader r;
		means_ifs.read(reinterpret_cast<char *>(&r), sizeof(r));
		assert(r.num == bh.biHeight * bh.biWidth);

		DPIXEL *cent = new DPIXEL[r.ncent];
		BITMAPPIXEL *bcent = new BITMAPPIXEL[r.ncent];

		means_ifs.read(reinterpret_cast<char *>(cent), sizeof(DPIXEL)*r.ncent);
		for (int i = 0; i < r.ncent; ++i)
		{
			bcent[i].b = clip_8b(cent[i][0] * 255);
			bcent[i].g = clip_8b(cent[i][1] * 255);
			bcent[i].r = clip_8b(cent[i][2] * 255);
		}
		delete[] cent;

		std::ofstream avgd_bmp_ofs(argv[3], std::ios::binary);
		avgd_bmp_ofs.write(reinterpret_cast<char *>(&fh), sizeof(fh));
		avgd_bmp_ofs.write(reinterpret_cast<char *>(&bh), sizeof(bh));
		avgd_bmp_ofs.seekp(fh.bOffBits);
		char skipwrite[skip] = {0};

		for (int i = 0; i < bh.biHeight; ++i)
		{
			for (int j = 0; j < bh.biWidth; ++j)
			{
				int centre;
				means_ifs.read(reinterpret_cast<char *>(&centre), sizeof(centre));
				avgd_bmp_ofs.write(reinterpret_cast<char *>(&bcent[centre]), sizeof(BITMAPPIXEL));
			}

			avgd_bmp_ofs.write(skipwrite, skip);
		}
		std::flush(avgd_bmp_ofs);

		delete[] bcent;
	}
}
