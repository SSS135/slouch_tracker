import type {
  TrainingEvent_Deserialize,
  TrainingResultResponse_Serialize,
} from '@generated/bindings';
import {
  createTrainingChannel,
  nativeClient,
  type NativeClient,
} from '../native/client';

export interface TrainingState {
  readonly running: boolean;
  readonly event: TrainingEvent_Deserialize | null;
  readonly result: TrainingResultResponse_Serialize | null;
  readonly error: Error | null;
  start(doCv?: boolean): Promise<TrainingResultResponse_Serialize>;
  cancel(): Promise<void>;
  reconcile(): Promise<void>;
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

export function createTrainingState(client: NativeClient = nativeClient): TrainingState {
  let running = $state(false);
  let event = $state<TrainingEvent_Deserialize | null>(null);
  let result = $state<TrainingResultResponse_Serialize | null>(null);
  let error = $state<Error | null>(null);

  return {
    get running() {
      return running;
    },
    get event() {
      return event;
    },
    get result() {
      return result;
    },
    get error() {
      return error;
    },
    async start(doCv = true) {
      if (running) {
        throw new Error('Training is already running.');
      }

      running = true;
      event = null;
      result = null;
      error = null;
      const channel = createTrainingChannel((nextEvent) => {
        event = nextEvent;
        if (nextEvent.type === 'completed') {
          result = nextEvent.result;
        } else if (nextEvent.type === 'failed') {
          error = new Error(nextEvent.error);
        }
      });

      try {
        result = await client.trainModels(doCv, channel);
        return result;
      } catch (cause) {
        error = asError(cause);
        throw cause;
      } finally {
        running = false;
      }
    },
    async cancel() {
      await client.cancelTraining();
    },
    async reconcile() {
      const status = await client.getTrainingStatus();
      running = status.running;
    },
  };
}
