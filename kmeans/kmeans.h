#ifndef KMEANS_H
#define KMEANS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//#pragma pack(push, 1)
struct Header
{
	int64_t num;
	int64_t dim;
};

struct RHeader
{
	int32_t ncent;
	int32_t dim;
	int64_t num;
};
//#pragma pack(pop)

#ifdef __cplusplus
}
#endif

#endif // KMEANS_H
