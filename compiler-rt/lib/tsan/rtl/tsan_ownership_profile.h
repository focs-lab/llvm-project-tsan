//===-- tsan_ownership_profile.h --------------------------------*- C++ -*-===//

#ifndef TSAN_OWNERSHIP_PROFILE_H
#define TSAN_OWNERSHIP_PROFILE_H

#include "tsan_defs.h"
#include "tsan_shadow.h"

namespace __tsan {

struct ThreadState;

extern bool ownership_profile_enabled;

void InitializeOwnershipProfile();
void OwnershipProfileThreadStart(ThreadState *thr);
void OwnershipProfileBeginAccess(ThreadState *thr, AccessType typ);
void OwnershipProfileRestartAccess(ThreadState *thr);
void OwnershipProfileAccess(ThreadState *thr, uptr addr,
                            RawShadow *shadow_mem, AccessType typ,
                            RawShadow current, bool contains_same_access,
                            bool note_eligible);
void OwnershipProfileSetOwner(ThreadState *thr, uptr addr, uptr size);
void OwnershipProfileResetRange(uptr addr, uptr size);
void OwnershipProfileSetFocusRange(uptr addr, uptr size);
void PrintOwnershipProfile(ThreadState *thr);

}  // namespace __tsan

#endif  // TSAN_OWNERSHIP_PROFILE_H
