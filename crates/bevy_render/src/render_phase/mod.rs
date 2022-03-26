mod draw;
mod draw_state;

use change_tracking_vec::ChangeTrackingVec;
pub use draw::*;
pub use draw_state::*;

use bevy_ecs::prelude::{Component, Query};

use copyless::VecHelper;
use rdst::{RadixKey, RadixSort};
use rdst::tuner::{Algorithm, Tuner, TuningParams};

/// A resource to collect and sort draw requests for specific [`PhaseItems`](PhaseItem).
#[derive(Component)]
pub struct RenderPhase<I: PhaseItem> {
    pub items: ChangeTrackingVec<I>,
    pub sorted_revision: usize,
}

impl<I: PhaseItem> Default for RenderPhase<I> {
    fn default() -> Self {
        let items = ChangeTrackingVec::new();
        let sorted_revision = items.revision();
        Self {
            items,
            sorted_revision,
        }
    }
}

pub struct RenderPhaseTuner;
impl Tuner for RenderPhaseTuner {
    #[inline]
    fn pick_algorithm(&self, p: &TuningParams, _: &[usize]) -> Algorithm {
        if p.input_len <= 256 {
            return Algorithm::Comparative;
        }

        return Algorithm::LrLsb;
    }
}

impl<I: PhaseItem + RadixKey + Copy> RenderPhase<I> {
    /// Adds a [`PhaseItem`] to this render phase.
    #[inline]
    pub fn add(&mut self, item: I) {
        self.items.inner_mut().alloc().init(item);
    }

    /// Sorts all of its [`PhaseItems`](PhaseItem).
    #[inline]
    pub fn sort(&mut self) {
        let rev = self.items.revision();
        if rev == self.sorted_revision {
            return;
        }

        self.items.radix_sort_unstable();
        self.sorted_revision = rev;
    }
}

impl<I: BatchedPhaseItem> RenderPhase<I> {
    /// Batches the compatible [`BatchedPhaseItem`]s of this render phase
    pub fn batch(&mut self) {
        // TODO: this could be done in-place
        let mut items = std::mem::take(&mut self.items);
        let mut items = items.drain(..);

        self.items.reserve(items.len());

        // Start the first batch from the first item
        if let Some(mut current_batch) = items.next() {
            // Batch following items until we find an incompatible item
            for next_item in items {
                if matches!(
                    current_batch.add_to_batch(&next_item),
                    BatchResult::IncompatibleItems
                ) {
                    // Store the completed batch, and start a new one from the incompatible item
                    self.items.push(current_batch);
                    current_batch = next_item;
                }
            }
            // Store the last batch
            self.items.push(current_batch);
        }
    }
}

/// This system sorts all [`RenderPhases`](RenderPhase) for the [`PhaseItem`] type.
pub fn sort_phase_system<I: PhaseItem + RadixKey + Copy>(
    mut render_phases: Query<&mut RenderPhase<I>>,
) {
    for mut phase in render_phases.iter_mut() {
        phase.sort();
    }
}

/// This system batches the [`PhaseItem`]s of all [`RenderPhase`]s of this type.
pub fn batch_phase_system<I: BatchedPhaseItem>(mut render_phases: Query<&mut RenderPhase<I>>) {
    for mut phase in render_phases.iter_mut() {
        phase.batch();
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::entity::Entity;

    use super::*;

    #[test]
    fn batching() {
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct TestPhaseItem {
            entity: Entity,
            batch_range: Option<BatchRange>,
        }
        impl RadixKey for TestPhaseItem {
            const LEVELS: usize = 4;

            #[inline]
            fn get_level(&self, level: usize) -> u8 {
                let s = 0.0f32.to_bits();
                let u = if s >> 31 == 1 { !s } else { s ^ (1 << 31) };

                (u >> (level * 8)) as u8
            }
        }
        impl PhaseItem for TestPhaseItem {
            fn draw_function(&self) -> DrawFunctionId {
                unimplemented!();
            }
        }
        impl EntityPhaseItem for TestPhaseItem {
            fn entity(&self) -> bevy_ecs::entity::Entity {
                self.entity
            }
        }
        impl BatchedPhaseItem for TestPhaseItem {
            fn batch_range(&self) -> &Option<BatchRange> {
                &self.batch_range
            }

            fn batch_range_mut(&mut self) -> &mut Option<BatchRange> {
                &mut self.batch_range
            }
        }
        let mut render_phase = RenderPhase::<TestPhaseItem>::default();
        let items = [
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(0, 5)),
            },
            // This item should be batched
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(5, 10)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(0, 5)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(10, 15)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(5, 10)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: None,
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(10, 15)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(20, 25)),
            },
            // This item should be batched
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(25, 30)),
            },
            // This item should be batched
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(30, 35)),
            },
        ];
        for item in items {
            render_phase.add(item);
        }
        render_phase.batch();
        let items_batched = [
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(0, 10)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(0, 5)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(10, 15)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(5, 10)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: None,
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(10, 15)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(20, 35)),
            },
        ];
        assert_eq!(&*render_phase.items, items_batched);
    }
}
