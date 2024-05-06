import debugFactory from 'debug'
import {type Config} from '../games/core/config'
import {type Network} from '../networks/nnet'
import {UniformNetwork} from '../networks/implementations/uniform'

import {EventEmitter} from 'events'
import {CoreNet} from '../networks/implementations/core'
import {ResNet} from '../networks/implementations/conv'

const debug = debugFactory('muzero:sharedstorage:module')

export class SharedStorage {
    public networkCount: number
    private readonly latestNetwork_: Network
    private readonly maxNetworks: number
    private readonly updatedNetworkEvent: EventEmitter

    /**
     *
     * @param config Network configuration needed for creation of Neural Networks
     * @param config.observationSize Length of observation tensors (input for the representation model h)
     * @param config.actionSpaceSize Length of the action tensors (partial input for the dynamics model g)
     */
    constructor(
        private readonly config: Config
    ) {
        this.latestNetwork_ = this.initialize()
        this.maxNetworks = 2
        this.updatedNetworkEvent = new EventEmitter()
        this.networkCount = -1
    }

    public initialize(): Network {
        const model = new ResNet(this.config.observationSize, this.config.actionSpace)
        return new CoreNet(model, this.config.lrInit, this.config.numUnrollSteps)
    }

    public uniformNetwork(): Network {
        // make uniform network: policy -> uniform, value -> 0, reward -> 0
        return new UniformNetwork(this.config.actionSpace)
    }

    public latestNetwork(): Network {
        debug(`Picked the latest network - training step ${this.networkCount}`)
        return this.networkCount >= 0 ? this.latestNetwork_ : this.uniformNetwork()
    }

    public async waitForUpdate(): Promise<Network> {
        const promise = new Promise<Network>((resolve, _) => {
            /*
            setTimeout(() => {
              this.updatedNetworkEvent.emit('update_event')
              reject('timed out')
            }, 10000)
             */
            this.updatedNetworkEvent.once('update_event', () => {
                resolve(this.latestNetwork())
            })
        })
        return await promise
    }

    public async loadNetwork(): Promise<void> {
        try {
            debug('Loading network')
            await this.latestNetwork_.load('file://data/'.concat(this.config.savedNetworkPath, '/'))
            this.networkCount = 0
        } catch (e) {
            debug(e)
        }
    }

    public async saveNetwork(step: number, network: Network): Promise<void> {
        debug('Saving network')
        await network.save('file://data/'.concat(this.config.savedNetworkPath, '/'))
        network.copyWeights(this.latestNetwork_)
        this.networkCount = step
        this.updatedNetworkEvent.emit('update_event')
    }
}
